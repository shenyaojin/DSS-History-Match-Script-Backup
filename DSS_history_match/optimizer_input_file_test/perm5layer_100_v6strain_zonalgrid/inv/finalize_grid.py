# Detached finalizer for the v6 sensitivity grid.
#
# Runs independently of any Claude conversation / tmux / terminal: launch it with
#   setsid nohup <pybin> finalize_grid.py > finalize_grid.log 2>&1 &
# so it is reparented to init and survives the conversation being closed.
#
# It polls until all 27 grid cells have produced optimized_theta_zonal.txt (or
# the scheduler process has exited), then runs compare_grid_results.py to write
# the final table + heatmaps + main-effects figure, drops a GRID_FINALIZED.txt
# marker, and exits.

import glob
import os
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PYBIN = "/rcp/rcp42/home/shenyaojin/miniforge/envs/moose/bin/python"
SCHEDULER_PID = 140782          # run_all_grid.sh; if it dies we finalize whatever exists
POLL_SECONDS = 300
TOTAL_CELLS = 27


def n_done():
    return sum(
        os.path.exists(os.path.join(d, "optimized_theta_zonal.txt"))
        for d in glob.glob(os.path.join(HERE, "bg*_w*_s*"))
    )


def scheduler_alive():
    return os.path.exists(f"/proc/{SCHEDULER_PID}")


def main():
    while True:
        done = n_done()
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  done={done}/{TOTAL_CELLS} "
              f"scheduler_alive={scheduler_alive()}", flush=True)
        if done >= TOTAL_CELLS:
            print("all cells done -> finalizing", flush=True)
            break
        if not scheduler_alive():
            print(f"scheduler {SCHEDULER_PID} gone at {done}/{TOTAL_CELLS} -> "
                  "finalizing partial", flush=True)
            break
        time.sleep(POLL_SECONDS)

    r = subprocess.run([PYBIN, "compare_grid_results.py"], cwd=HERE,
                       capture_output=True, text=True)
    print("=== compare stdout ===\n" + r.stdout, flush=True)
    if r.stderr:
        print("=== compare stderr ===\n" + r.stderr, flush=True)
    with open(os.path.join(HERE, "GRID_FINALIZED.txt"), "w") as f:
        f.write(f"finalized at {time.strftime('%Y-%m-%d %H:%M:%S')}  "
                f"done={n_done()}/{TOTAL_CELLS}  compare_rc={r.returncode}\n")
    print("finalizer done.", flush=True)


if __name__ == "__main__":
    main()
