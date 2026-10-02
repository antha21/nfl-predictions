import subprocess

print("--- STARTING WEEKLY NFL RESET ---")
subprocess.run(["python", "build_nfl_data.py"])
subprocess.run(["python", "calculate_fantasy_points.py"])
print("--- WEEKLY RESET COMPLETE! READY FOR PROJECTIONS ---")