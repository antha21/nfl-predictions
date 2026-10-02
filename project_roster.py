"""
project_roster.py  (V8)

Fixes vs V7
  * Players are keyed by gsis player_id, never by display name. Name keys merged different
    players that share a name and split one player across name variants ("M.Wilson" / "Mi.Wilson"
    are the same Michael Wilson id), which is how players went missing.
  * Projections start from the CURRENT depth chart (latest snapshot) and the CURRENT team, so
    linebackers/free agents/practice-squad guys with a stray fumble no longer get projections.
  * Availability = manual override > roster status (IR/PUP/NFI/SUS/...) > this week's report
    > multi-week "Out" streak > "starter missed last 3 team games with no report".
  * Rolling baselines now INCLUDE the player's most recent game (V7 used a shift(1) feature
    row, so every baseline was one game stale). Games on/after the upcoming week are excluded.
  * Succession logic no longer copies the starter's line onto every backup (identical 25.95s).
  * Kickers (own model: form, opposing-K allowed, implied team points, total, wind/temp/dome)
    and a Vegas-based D/ST heuristic are exported.
  * No blanket `except: continue`. Every omission is written to an audit CSV with a reason.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
import nflreadpy as nfl
import warnings
warnings.filterwarnings('ignore')

UPCOMING_SEASON = 2026
UPCOMING_WEEK = 4

# Force a status when news breaks before the feeds catch up.
# Values: Out, Ir, Pup, Doubtful, Questionable, or Active (clears a stale flag).
MANUAL_OVERRIDES = {
    'Jordan Mason': 'Ir',       # thumb -> IR (reported by user; remove once feeds catch up)
}

# Kickoff weather isn't in nflverse schedules until after a game. Fill in forecasts by HOME team.
WEATHER_OVERRIDES = {
    # 'GB': {'temp': 28, 'wind': 18},
}

WATCHLIST = ['Michael Wilson', 'Jordan Mason', 'Aaron Jones']   # printed in the audit summary

# ---- Tuning ------------------------------------------------------------------------------
SKILL = ['QB', 'RB', 'WR', 'TE']
STARTERS = {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1}               # effective ranks that skip the usage test
DEPTH_CAP = {'QB': 2, 'RB': 4, 'WR': 6, 'TE': 3}              # deeper than this is never projected
MIN_USAGE = {'QB': 8.0, 'RB': 4.0, 'WR': 2.5, 'TE': 2.0}      # avg last-3: attempts / carries+targets / targets
PROMO = {'QB': 0.80, 'RB': 0.90, 'TE': 0.80}                  # successor floor = factor * vacated baseline
WR_SPREAD = 0.15                                              # each of top-3 healthy WRs gains 15% of a vacated WR
QUESTIONABLE_MULT = 0.85
LONG_TERM_OUT_WEEKS = 2                                       # Out this many straight weeks => long-term
STALE_GAMES = 3                                               # starter missing all of last N team games
INACTIVE_ROSTER = {'RES', 'PUP', 'NFI', 'NWT', 'SUS', 'CUT', 'TRC', 'DEV', 'EXE', 'UDF', 'DRT', 'RET', 'SUSP'}
OUT_REPORT = {'Out', 'Ir', 'Injured Reserve', 'Doubtful', 'Pup', 'Nfi', 'Suspended', 'Sus'}

# Kicker scoring (edit to match your league)
K_FG_0_39, K_FG_40_49, K_FG_50P, K_PAT, K_FG_MISS, K_PAT_MISS = 3.0, 4.0, 5.0, 1.0, -1.0, -1.0

TEAM_FIX = {'ARZ': 'ARI', 'BLT': 'BAL', 'CLV': 'CLE', 'HST': 'HOU', 'JAC': 'JAX', 'LAR': 'LA',
            'OAK': 'LV', 'SD': 'LAC', 'SL': 'LA', 'STL': 'LA', 'WSH': 'WAS'}

SKILL_FEATS = ['p_roll_pts_adj', 'p_roll_rush_adj', 'p_roll_rec_adj', 'p_roll_pass_adj',
               'd_roll_pts', 'd_roll_rush', 'd_roll_rec', 'd_roll_pass', 'team_spread', 'total_line']
K_FEATS = ['k_roll_pts', 'opp_k_allowed', 'implied', 'total_line', 'wind', 'temp', 'is_dome']
ADJ = ['adj_pts', 'adj_rush', 'adj_rec', 'adj_pass']


def fix_team(t):
    return TEAM_FIX.get(t, t) if isinstance(t, str) else t


def col(df, name):
    return df[name].fillna(0) if name in df.columns else pd.Series(0.0, index=df.index)


def kicker_points(s):
    return ((col(s, 'fg_made_0_19') + col(s, 'fg_made_20_29') + col(s, 'fg_made_30_39')) * K_FG_0_39
            + col(s, 'fg_made_40_49') * K_FG_40_49
            + (col(s, 'fg_made_50_59') + col(s, 'fg_made_60_')) * K_FG_50P
            + col(s, 'pat_made') * K_PAT + col(s, 'fg_missed') * K_FG_MISS + col(s, 'pat_missed') * K_PAT_MISS)


def latest_snapshot(df, season, week):
    if 'season' in df.columns:
        cur = df[df['season'] == season]
        df = cur if not cur.empty else df
    if 'dt' in df.columns and df['dt'].notna().any():
        return df[df['dt'] == df['dt'].max()]
    if 'week' in df.columns:
        wk = df['week'][df['week'] <= week]
        return df[df['week'] == (wk.max() if len(wk) else df['week'].max())]
    return df


def generate_weekly_projections():
    cutoff = lambda d: d[(d['season'] < UPCOMING_SEASON) | ((d['season'] == UPCOMING_SEASON) & (d['week'] < UPCOMING_WEEK))]
    audit = []

    def skip(pid, name, team, pos, reason):
        audit.append({'Player ID': pid, 'Player': name, 'Team': team, 'Position': pos, 'Reason': reason})

    print("1. Loading history + nflverse feeds...")
    scores = pd.read_csv('historical_fantasy_scores.csv')
    # Collapse name variants: one row per player_id per week (player_name is NOT part of the key).
    num = ['total_pass_pts', 'total_rush_pts', 'total_rec_pts', 'total_fum_pts', 'weekly_fantasy_points']
    scores = scores.groupby(['season', 'week', 'player_id'], as_index=False)[num].sum()

    stats = nfl.load_player_stats(range(2021, UPCOMING_SEASON + 1)).to_pandas()
    stats['team'] = stats['team'].map(fix_team)
    stats['opponent_team'] = stats['opponent_team'].map(fix_team)
    stats = stats.drop_duplicates(['season', 'week', 'player_id'])
    stats = cutoff(stats)
    name_by_id = stats.sort_values(['season', 'week']).groupby('player_id')['player_display_name'].last().to_dict()

    keep = [c for c in ['season', 'week', 'player_id', 'team', 'opponent_team', 'player_display_name', 'position',
                        'carries', 'targets', 'attempts'] if c in stats.columns]
    df = cutoff(scores).merge(stats[keep], on=['season', 'week', 'player_id'], how='inner')
    df = df[df['position'].isin(SKILL)]                      # drops LBs/DBs/specialists with stray plays
    for c in ['carries', 'targets', 'attempts']:
        df[c] = df[c].fillna(0) if c in df.columns else 0.0

    # ---- Schedule / Vegas / weather game table ---------------------------------------------
    sched = nfl.load_schedules(range(2021, UPCOMING_SEASON + 1)).to_pandas()
    sched['home_team'] = sched['home_team'].map(fix_team)
    sched['away_team'] = sched['away_team'].map(fix_team)
    roof = sched['roof'] if 'roof' in sched.columns else pd.Series('', index=sched.index)
    sched['is_dome'] = roof.isin(['dome', 'closed']).astype(float)
    for c in ['temp', 'wind']:
        if c not in sched.columns:
            sched[c] = np.nan
    sched.loc[sched['is_dome'] == 1, ['temp', 'wind']] = [70.0, 0.0]
    for home, w in WEATHER_OVERRIDES.items():
        m = (sched['season'] == UPCOMING_SEASON) & (sched['week'] == UPCOMING_WEEK) & (sched['home_team'] == home)
        for k, v in w.items():
            sched.loc[m, k] = v
    base = ['season', 'week', 'spread_line', 'total_line', 'temp', 'wind', 'is_dome']
    h = sched[base + ['home_team', 'away_team']].rename(columns={'home_team': 'team', 'away_team': 'opp'})
    h['team_spread'] = -h['spread_line']
    a = sched[base + ['away_team', 'home_team']].rename(columns={'away_team': 'team', 'home_team': 'opp'})
    a['team_spread'] = a['spread_line']
    games = pd.concat([h, a]).dropna(subset=['team_spread', 'total_line'])
    games['implied'] = games['total_line'] / 2 - games['team_spread'] / 2
    gcols = ['season', 'week', 'team', 'team_spread', 'total_line', 'implied', 'temp', 'wind', 'is_dome']

    up_sched = sched[(sched['season'] == UPCOMING_SEASON) & (sched['week'] == UPCOMING_WEEK)]
    matchups = dict(zip(up_sched['home_team'], up_sched['away_team']))
    matchups.update(dict(zip(up_sched['away_team'], up_sched['home_team'])))
    up_games = {r['team']: r for _, r in games[(games['season'] == UPCOMING_SEASON) & (games['week'] == UPCOMING_WEEK)].iterrows()}

    # ---- Skill-position features -----------------------------------------------------------
    print("2. Training skill-position model...")
    df = df.merge(games[gcols], on=['season', 'week', 'team'], how='inner').sort_values(['player_id', 'season', 'week'])
    dcols = ['weekly_fantasy_points', 'total_rush_pts', 'total_rec_pts', 'total_pass_pts']
    def_stats = df.groupby(['season', 'week', 'opponent_team'])[dcols].sum().reset_index().sort_values(['opponent_team', 'season', 'week'])
    for src, dst in zip(dcols, ['d_roll_pts', 'd_roll_rush', 'd_roll_rec', 'd_roll_pass']):
        def_stats[dst] = def_stats.groupby(['opponent_team', 'season'])[src].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    df = df.merge(def_stats[['season', 'week', 'opponent_team', 'd_roll_pts', 'd_roll_rush', 'd_roll_rec', 'd_roll_pass']],
                  on=['season', 'week', 'opponent_team'], how='left')
    lg = def_stats.groupby('season')['d_roll_pts'].mean().rename('league_avg_d_pts').reset_index()
    df = df.merge(lg, on='season', how='left')
    df['sos'] = np.where(df['d_roll_pts'] > 0, df['league_avg_d_pts'] / df['d_roll_pts'], 1.0)
    df['sos'] = pd.Series(df['sos']).fillna(1.0).clip(0.7, 1.3).values
    for src, dst in zip(dcols, ADJ):
        df[dst] = df[src] * df['sos']
    for dst, src in zip(['p_roll_pts_adj', 'p_roll_rush_adj', 'p_roll_rec_adj', 'p_roll_pass_adj'], ADJ):
        df[dst] = df.groupby('player_id')[src].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())

    train = df.dropna(subset=SKILL_FEATS)
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=4)
    model.fit(train[SKILL_FEATS], train['weekly_fantasy_points'])

    # Current baselines INCLUDE the latest game
    last3 = df.groupby('player_id').tail(3)
    base_by_id = last3.groupby('player_id')[ADJ].mean()
    use_by_id = last3.groupby('player_id')[['carries', 'targets', 'attempts']].mean()
    d_recent = def_stats.groupby('opponent_team').tail(5).groupby('opponent_team')[dcols].mean()
    d_league = d_recent.mean()
    player_weeks = df[df['season'] == UPCOMING_SEASON].groupby('player_id')['week'].apply(set).to_dict()
    team_weeks = df[df['season'] == UPCOMING_SEASON].groupby('team')['week'].apply(lambda s: sorted(set(s))).to_dict()

    # ---- Kicker model ----------------------------------------------------------------------
    print("3. Training kicker model...")
    kst = stats[stats['position'] == 'K'].copy()
    kst['k_pts'] = kicker_points(kst)
    kst = kst.merge(games[gcols], on=['season', 'week', 'team'], how='inner').sort_values(['player_id', 'season', 'week'])
    k_allowed = kst.groupby(['season', 'week', 'opponent_team'])['k_pts'].sum().reset_index().sort_values(['opponent_team', 'season', 'week'])
    k_allowed['opp_k_allowed'] = k_allowed.groupby(['opponent_team', 'season'])['k_pts'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    kst = kst.merge(k_allowed[['season', 'week', 'opponent_team', 'opp_k_allowed']], on=['season', 'week', 'opponent_team'], how='left')
    kst['k_roll_pts'] = kst.groupby('player_id')['k_pts'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    ktrain = kst.dropna(subset=['k_roll_pts', 'opp_k_allowed', 'implied', 'total_line'])
    k_model = None
    if len(ktrain) > 100:
        k_model = xgb.XGBRegressor(n_estimators=60, learning_rate=0.08, max_depth=3)
        k_model.fit(ktrain[K_FEATS], ktrain['k_pts'])     # NaN weather is handled natively by XGBoost
    else:
        print("   WARNING: not enough kicker rows; kickers will use form + implied-points fallback.")
    k_form = kst.groupby('player_id').tail(3).groupby('player_id')['k_pts'].mean()
    k_opp = k_allowed.groupby('opponent_team').tail(5).groupby('opponent_team')['k_pts'].mean()
    k_games = set(kst[kst['season'] == UPCOMING_SEASON]['player_id'])

    # ---- Depth chart, rosters, injuries ----------------------------------------------------
    print(f"4. Pulling Week {UPCOMING_WEEK} depth chart, roster status and injury data...")
    dc = nfl.load_depth_charts([UPCOMING_SEASON]).to_pandas()
    dc = latest_snapshot(dc, UPCOMING_SEASON, UPCOMING_WEEK)
    dc = dc[dc['pos_abb'].isin(SKILL + ['K']) & dc['gsis_id'].notna()].copy()
    dc['team'] = dc['team'].map(fix_team)
    dc['pos_rank'] = pd.to_numeric(dc['pos_rank'], errors='coerce').fillna(99)
    pos_by_id = stats.sort_values(['season', 'week']).groupby('player_id')['position'].last().to_dict()
    dc['pref'] = (dc['pos_abb'] != dc['gsis_id'].map(pos_by_id)).astype(int)
    dc = dc.sort_values(['pref', 'pos_rank']).drop_duplicates('gsis_id')   # one primary position per player

    roster_status = {}
    try:
        rw = nfl.load_rosters_weekly([UPCOMING_SEASON]).to_pandas()
        rw = latest_snapshot(rw, UPCOMING_SEASON, UPCOMING_WEEK)
        for _, r in rw.iterrows():
            st = str(r.get('status', '')).upper()
            desc = str(r.get('status_description_abbr', '')).upper()
            if st in INACTIVE_ROSTER or (desc.startswith(('R0', 'P0', 'N0', 'S0')) and desc != 'A01'):
                roster_status[r['gsis_id']] = f"{st}/{desc}"
    except Exception as e:
        print(f"   WARNING: roster status feed failed ({e}). IR detection falls back to injury reports.")

    inj_now, inj_hist = {}, {}
    try:
        inj = nfl.load_injuries([UPCOMING_SEASON]).to_pandas()
        inj['report_status'] = inj['report_status'].astype(str).str.strip().str.title()
        for _, r in inj[inj['week'] == UPCOMING_WEEK].iterrows():
            inj_now[r['gsis_id']] = (r['report_status'], str(r.get('report_primary_injury', '') or ''))
        prev = inj[(inj['week'] < UPCOMING_WEEK) & (inj['week'] >= UPCOMING_WEEK - LONG_TERM_OUT_WEEKS)]
        for pid, g in prev[prev['report_status'].isin(OUT_REPORT)].groupby('gsis_id'):
            inj_hist[pid] = g['week'].nunique()
    except Exception as e:
        print(f"   WARNING: injury feed failed ({e}).")

    def availability(pid, name, team, pos, depth_rank):
        """-> (state, tag). state is 'OUT', 'Q' or 'OK'."""
        ov = MANUAL_OVERRIDES.get(name) or MANUAL_OVERRIDES.get(pid)
        if ov:
            ov = ov.strip().title()
            if ov == 'Active':
                return 'OK', ''
            if ov == 'Questionable':
                return 'Q', '[QUESTIONABLE] '
            return 'OUT', f'[{ov.upper()}] '
        if pid in roster_status:
            return 'OUT', f'[IR/INACTIVE: {roster_status[pid]}] '
        status, body = inj_now.get(pid, ('', ''))
        if status in OUT_REPORT:
            multi = inj_hist.get(pid, 0) >= LONG_TERM_OUT_WEEKS - 1
            return 'OUT', f"[{'OUT MULTI-WEEK' if multi else status.upper()}{': ' + body if body else ''}] "
        if status == 'Questionable':
            return 'Q', '[QUESTIONABLE] '
        if inj_hist.get(pid, 0) >= LONG_TERM_OUT_WEEKS:
            return 'OUT', '[OUT LAST 2 WEEKS, NO UPDATE] '
        recent = team_weeks.get(team, [])[-STALE_GAMES:]
        if pos in STARTERS and depth_rank <= STARTERS[pos] and len(recent) >= STALE_GAMES \
                and not (set(recent) & player_weeks.get(pid, set())):
            return 'OUT', f'[NO GAMES LAST {STALE_GAMES} TEAM WEEKS] '
        return 'OK', ''

    # ---- Build candidates from the depth chart ---------------------------------------------
    print("5. Building candidate pool from depth chart...")
    cands = []
    for _, r in dc.iterrows():
        pid, pos, team = r['gsis_id'], r['pos_abb'], r['team']
        name = name_by_id.get(pid, r.get('player_name', pid))
        if pos == 'K':
            if pid not in k_form.index:
                skip(pid, name, team, pos, 'kicker has no historical kicking rows'); continue
            cands.append({'id': pid, 'name': name, 'pos': 'K', 'team': team, 'rank': r['pos_rank'], 'base': np.array([k_form[pid], 0, 0, 0.0])})
            continue
        if pid not in base_by_id.index:
            skip(pid, name, team, pos, 'on depth chart but no regular-season history (rookie/new) - cannot project'); continue
        if r['pos_rank'] > DEPTH_CAP[pos]:
            skip(pid, name, team, pos, f'depth rank {int(r["pos_rank"]) if r["pos_rank"] < 99 else "n/a"} beyond cap {DEPTH_CAP[pos]}'); continue
        cands.append({'id': pid, 'name': name, 'pos': pos, 'team': team, 'rank': r['pos_rank'], 'base': base_by_id.loc[pid].values.astype(float)})

    groups = {}
    for c in cands:
        groups.setdefault((c['team'], c['pos']), []).append(c)
    for (team, pos), g in groups.items():
        g.sort(key=lambda c: (c['rank'], -c['base'][0]))           # re-rank sequentially (ties/slots in feed)
        for i, c in enumerate(g):
            c['depth'] = i + 1
            c['state'], c['tag'] = availability(c['id'], c['name'], team, pos, c['depth'])

    # ---- Next-man-up: reassign vacated volume ----------------------------------------------
    for (team, pos), g in groups.items():
        if pos == 'K':
            continue
        avail = [c for c in g if c['state'] != 'OUT']
        for i, c in enumerate(avail):
            c['eff'] = i + 1
        vacated = sorted([c for c in g if c['state'] == 'OUT' and c['depth'] <= 3 and c['base'][0] >= 3],
                         key=lambda c: -c['base'][0])
        used = set()
        for v in vacated:
            if pos in PROMO:
                pool = [c for c in avail[:2 if pos == 'RB' else 1] if c['id'] not in used]
                if not pool:
                    continue
                s = pool[0]
                new = np.maximum(s['base'], PROMO[pos] * v['base'])
                if (new > s['base']).any():
                    s['base'] = new
                    s['tag'] += f"[NEXT MAN UP: {v['name']}] "
                used.add(s['id'])
            else:  # WR: spread targets across the healthy top 3
                for s in avail[:3]:
                    s['base'] = s['base'] + WR_SPREAD * v['base']
                    s['tag'] += f"[TARGETS UP: {v['name']} OUT] "
    qb_out_teams = {t for (t, p), g in groups.items() if p == 'QB' and any(c['depth'] == 1 and c['state'] == 'OUT' for c in g)}

    # ---- Project ---------------------------------------------------------------------------
    print(f"\n--- WEEK {UPCOMING_WEEK} V8 PROJECTIONS ---")
    rows = []

    def row(c, proj, opp, g, note, extra=''):
        rows.append({'Player': c['name'], 'Position': c['pos'], 'Team': c['team'], 'Opponent': opp,
                     'Spread': g['team_spread'] if g is not None else '', 'Total': g['total_line'] if g is not None else '',
                     'Projected Points': round(float(proj), 2), 'Status / Notes': note.strip(),
                     'Player ID': c['id'], 'Depth': c.get('depth', ''),
                     'Implied Pts': round(float(g['implied']), 1) if g is not None else '', 'Weather': extra})

    for c in cands:
        try:
            team, pos = c['team'], c['pos']
            opp = matchups.get(team)
            if pos != 'K':
                u = use_by_id.loc[c['id']]
                usage = {'QB': u['attempts'], 'RB': u['carries'] + u['targets'], 'WR': u['targets'], 'TE': u['targets']}[pos]
                relevant = c.get('eff', 99) <= STARTERS[pos] or usage >= MIN_USAGE[pos] or c['depth'] <= 1
                if c['state'] == 'OUT':
                    if c['depth'] > STARTERS[pos] + 1 and c['base'][0] < 3:
                        skip(c['id'], c['name'], team, pos, f"unavailable and low-impact ({c['tag'].strip()})"); continue
                elif not relevant:
                    skip(c['id'], c['name'], team, pos, f'below usage threshold ({usage:.1f} < {MIN_USAGE[pos]})'); continue
            elif c['state'] != 'OUT' and c['depth'] > 1:
                skip(c['id'], c['name'], team, pos, 'not the primary kicker'); continue

            if c['state'] == 'OUT':
                row(c, 0.0, opp or 'BYE', up_games.get(team), c['tag'] or '[OUT] '); continue
            if not opp:
                row(c, 0.0, 'BYE', None, '[BYE] '); continue
            g = up_games.get(team)
            if g is None:
                skip(c['id'], c['name'], team, pos, 'no Vegas line for this game yet'); continue

            mult = QUESTIONABLE_MULT if c['state'] == 'Q' else 1.0
            note = c['tag']
            if pos == 'K':
                opp_k = k_opp.get(opp, k_opp.mean())
                if k_model is not None:
                    X = pd.DataFrame([[c['base'][0], opp_k, g['implied'], g['total_line'], g['wind'], g['temp'], g['is_dome']]], columns=K_FEATS)
                    proj = k_model.predict(X)[0]
                else:
                    proj = 0.5 * c['base'][0] + 0.5 * (g['implied'] / 22.0) * 8.0
                wx = 'dome' if g['is_dome'] == 1 else (f"{g['temp']:.0f}F/{g['wind']:.0f}mph" if pd.notna(g['wind']) and pd.notna(g['temp']) else 'wx n/a')
                row(c, max(proj, 0) * mult, opp, g, note, wx); continue

            p = c['base'].copy()
            if team in qb_out_teams and (pos in ('WR', 'TE') or (pos == 'RB' and p[2] > 0.5 * p[0])):
                p[0] *= 0.75; p[2] *= 0.75; note += '[BACKUP QB PENALTY] '
            d = d_recent.loc[opp] if opp in d_recent.index else d_league
            X = pd.DataFrame([[p[0], p[1], p[2], p[3], d['weekly_fantasy_points'], d['total_rush_pts'], d['total_rec_pts'],
                               d['total_pass_pts'], g['team_spread'], g['total_line']]], columns=SKILL_FEATS)
            row(c, max(model.predict(X)[0], 0) * mult, opp, g, note)
        except Exception as e:
            skip(c['id'], c['name'], c['team'], c['pos'], f'ERROR: {e}')

    # ---- D/ST (Vegas heuristic, no defensive stats in this pipeline) -----------------------
    for team, g in up_games.items():
        opp = matchups.get(team)
        og = up_games.get(opp)
        if og is None:
            continue
        proj = float(np.clip(8.5 - 0.40 * (og['implied'] - 22.0) - 0.12 * g['team_spread'], 1.0, 16.0))
        row({'name': f'{team} D/ST', 'pos': 'D/ST', 'team': team, 'id': f'DST_{team}'}, proj, opp, g, '[VEGAS HEURISTIC] ')

    out = pd.DataFrame(rows).sort_values('Projected Points', ascending=False)
    fn = f"nfl_projections_week_{UPCOMING_WEEK}.csv"
    out.to_csv(fn, index=False)
    pd.DataFrame(audit).to_csv(f"nfl_projection_audit_week_{UPCOMING_WEEK}.csv", index=False)
    print(f"Success! {len(out)} rows -> {fn}  (position counts: {out['Position'].value_counts().to_dict()})")
    print(f"Omitted {len(audit)} depth-chart players, reasons in nfl_projection_audit_week_{UPCOMING_WEEK}.csv")

    print("\n--- WATCHLIST ---")
    for w in WATCHLIST:
        hit = out[out['Player'].str.contains(w, case=False, na=False)]
        miss = [x for x in audit if w.lower() in str(x['Player']).lower()]
        if not hit.empty:
            print(hit[['Player', 'Position', 'Team', 'Projected Points', 'Status / Notes']].to_string(index=False, header=False))
        for m in miss:
            print(f"  {m['Player']} OMITTED: {m['Reason']}")
        if hit.empty and not miss:
            print(f"  {w}: not on the current depth chart - check the name, team and nfl.load_depth_charts output.")


if __name__ == '__main__':
    generate_weekly_projections()