# static_train.py
# Implements a static traffic light controller with a fixed cycle time
# for comparison against a dynamic model.

from __future__ import absolute_import, print_function

import os
import sys
import optparse
import random
import numpy as np
import torch # Kept torch/nn/optim imports for minimal change, though not used in static control
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary
import traci

# --- Constants from train.py ---
DEFAULT_TRAIN_CFG = "pune.sumocfg"
DEFAULT_TEST_CFG = "pune_test.sumocfg"
TRIPINFO_OUTPUT = "maps/tripinfo.xml"
PLOTS_DIR = "plots" # Only PLOTS_DIR is needed, MODELS_DIR is removed
DEFAULT_MAX_VEHICLES_PER_LANE = 0.0 # Not used but kept for context

# --- Utility Functions (Simplified/Kept) ---

def ensure_dirs():
    # Only need PLOTS_DIR and maps folder
    os.makedirs(PLOTS_DIR, exist_ok=True)
    maps_folder = os.path.dirname(TRIPINFO_OUTPUT)
    if maps_folder:
        os.makedirs(maps_folder, exist_ok=True)

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    # torch seeds are not needed for static, but kept for consistency
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_vehicle_numbers(lanes):
    # Kept for compatibility with diagnostics if needed later
    vehicle_per_lane = []
    for l in lanes:
        cnt = 0
        for veh_id in traci.lane.getLastStepVehicleIDs(l):
            try:
                if traci.vehicle.getLanePosition(veh_id) > 10:
                    cnt += 1
            except Exception:
                pass
        vehicle_per_lane.append(cnt)
    return vehicle_per_lane

def get_waiting_time(lanes):
    # Kept for compatibility with reward shaping/diagnostics if needed later
    waiting_time = 0.0
    for lane in lanes:
        waiting_time += traci.lane.getWaitingTime(lane)
    return waiting_time

def phaseDuration(junction, phase_time, phase_state):
    # Helper to set traffic light phase and duration
    traci.trafficlight.setRedYellowGreenState(junction, phase_state)
    traci.trafficlight.setPhaseDuration(junction, phase_time)

# Removed Model and Agent classes as they are for dynamic control

# --- Main Run Function (Modified for Static Control) ---

def run_static(model_name="static_model", epochs=50, steps=500, cfg=None, nogui=False, seed=42):
    ensure_dirs()
    set_seeds(seed)

    cfg = cfg or DEFAULT_TEST_CFG # Use the test cfg by default for comparison

    # --- Static Traffic Light Parameters ---
    GREEN_DURATION = 60 # Main green phase duration in seconds
    YELLOW_DURATION = 3 # Yellow phase duration
    ALL_RED_DURATION = 3 # All red phase duration
    CYCLE_DURATION = YELLOW_DURATION + ALL_RED_DURATION # Total transition time between main greens

    # Note: The original 'select_lane' phases define a 4-way intersection pattern.
    # The patterns are: Yellow/Red -> Green -> Yellow/Red -> ...
    # select_lane[i][0] is the transition phase (e.g., yellow)
    # select_lane[i][1] is the main phase (e.g., green)
    select_lane = [
        ["yyyrrrrrrrrr", "GGGrrrrrrrrr"], # Phase 0 (Main Street A Green)
        ["rrryyyrrrrrr", "rrrGGGrrrrrr"], # Phase 1 (Main Street B Green)
        ["rrrrrryyyrrr", "rrrrrrGGGrrr"], # Phase 2 (Side Street A Green)
        ["rrrrrrrrryyy", "rrrrrrrrrGGG"], # Phase 3 (Side Street B Green)
    ]
    N_PHASES = len(select_lane)

    avg_wait_list = []

    for e in range(epochs):
        # Start SUMO simulation
        sumo_bin = checkBinary("sumo") if nogui else checkBinary("sumo-gui")
        seed_arg = ["--seed", str(seed + e)]
        try:
            traci.start([sumo_bin, "-c", cfg, "--tripinfo-output", TRIPINFO_OUTPUT] + seed_arg)
        except Exception:
            try:
                traci.close()
            except:
                pass
            traci.start([sumo_bin, "-c", cfg, "--tripinfo-output", TRIPINFO_OUTPUT])

        all_junctions = traci.trafficlight.getIDList()
        if len(all_junctions) == 0:
            print("Warning: no traffic lights in cfg", cfg)
            traci.close()
            continue

        # Initial state for static control
        current_phase_idx = {j: 0 for j in all_junctions}
        phase_time_left = {j: 0 for j in all_junctions} # Timer for the current phase

        veh_waits = {}    # vid -> last seen accumulated waiting time (sec)
        arrived = set()   # finished vehicle ids
        seen_vehicles = set()
        teleport_ids = set()

        step = 0
        while step <= steps:
            traci.simulationStep()

            # --- Diagnostics and Wait Time Collection (Same as dynamic model) ---
            try:
                current_vehicle_ids = traci.vehicle.getIDList()
                seen_vehicles.update(current_vehicle_ids)
                for vid in current_vehicle_ids:
                    try:
                        veh_waits[vid] = traci.vehicle.getAccumulatedWaitingTime(vid)
                    except Exception:
                        pass
            except Exception:
                pass

            try:
                arrived_ids = traci.simulation.getArrivedIDList()
                if arrived_ids:
                    arrived.update(arrived_ids)
            except Exception:
                pass

            try:
                started_tele = traci.simulation.getStartingTeleportIDList()
                ended_tele = traci.simulation.getEndingTeleportIDList()
                if started_tele:
                    teleport_ids.update(started_tele)
                if ended_tele:
                    teleport_ids.update(ended_tele)
            except Exception:
                pass
            # --- End of Diagnostics ---


            for junction in all_junctions:
                if phase_time_left[junction] <= 0:
                    # Time to transition to the next main phase
                    current_phase_idx[junction] = (current_phase_idx[junction] + 1) % N_PHASES
                    next_main_phase_idx = current_phase_idx[junction]

                    # 1. Apply the yellow/transition phase from the *current* main phase's setting
                    # This phase must be applied before the next green, but for simplicity
                    # we use a full all-red and yellow state transition.
                    # A common static TL logic is: Green -> Yellow -> All-Red -> Next Green

                    # Apply transition phase (e.g., Yellow from the old phase)
                    # For simplicity and to match the 'select_lane' structure where
                    # index [0] is yellow/transition and [1] is green:

                    # Apply Yellow/Transition phase (select_lane[current_phase_idx][0])
                    phaseDuration(junction, YELLOW_DURATION, select_lane[next_main_phase_idx][0])
                    phase_time_left[junction] = YELLOW_DURATION
                    traci.simulationStep() # Advance one step to execute phase change

                    # Apply All-Red phase (optional, but good practice. Not explicitly in select_lane,
                    # so we must define an all-red state, e.g., "rrrrrrrrrrrr")
                    # phaseDuration(junction, ALL_RED_DURATION, "rrrrrrrrrrrr")
                    # phase_time_left[junction] += ALL_RED_DURATION # Total transition time
                    # traci.simulationStep() # Advance one step to execute phase change

                    # Apply Green phase (select_lane[current_phase_idx][1])
                    phaseDuration(junction, GREEN_DURATION, select_lane[next_main_phase_idx][1])
                    phase_time_left[junction] += GREEN_DURATION
                else:
                    phase_time_left[junction] -= 1

            step += 1

        # --- diagnostics & robust averages (Same as dynamic model) ---
        all_vals = list(veh_waits.values())
        finished_vals = [veh_waits[v] for v in arrived if v in veh_waits]
        finished_no_tele = [veh_waits[v] for v in arrived if (v in veh_waits and v not in teleport_ids)]

        def stats(arr):
            if not arr:
                return {"count":0,"mean":0.0,"min":0.0,"median":0.0,"p75":0.0,"p90":0.0,"max":0.0,"zeros":0}
            a = np.array(arr, dtype=float)
            return {
                "count": len(a),
                "mean": float(np.mean(a)),
                "min": float(np.min(a)),
                "median": float(np.median(a)),
                "p75": float(np.percentile(a,75)),
                "p90": float(np.percentile(a,90)),
                "max": float(np.max(a)),
                "zeros": int((a==0).sum())
            }

        s_all = stats(all_vals)
        s_finished = stats(finished_vals)
        s_finished_no_tele = stats(finished_no_tele)

        # parse tripinfo.xml (if present) for SUMO's official waitingTime per trip
        tripinfo_stats = None
        try:
            import xml.etree.ElementTree as ET
            if os.path.exists(TRIPINFO_OUTPUT):
                tree = ET.parse(TRIPINFO_OUTPUT)
                root = tree.getroot()
                trip_waits = []
                for ti in root.findall('tripinfo'):
                    wt = float(ti.attrib.get('waitingTime', '0'))
                    trip_waits.append(wt)
                if trip_waits:
                    tripinfo_stats = {
                        "count": len(trip_waits),
                        "mean": float(np.mean(trip_waits)),
                        "median": float(np.median(trip_waits)),
                        "p75": float(np.percentile(trip_waits,75)),
                        "p90": float(np.percentile(trip_waits,90)),
                        "max": float(np.max(trip_waits))
                    }
        except Exception:
            tripinfo_stats = None

        # use 'all' mean as the metric for plotting
        metric_to_use = s_all["mean"]

        # print diagnostics
        print(f"STATIC run {e+1}/{epochs} (cfg={cfg}) Green={GREEN_DURATION}s")
        print(f"  all: count={s_all['count']} mean={s_all['mean']:.2f} sec ({s_all['mean']/60:.2f} min) "
              f"median={s_all['median']:.2f} p90={s_all['p90']:.2f} zeros={s_all['zeros']}")
        print(f"  finished (incl teleports): count={s_finished['count']} mean={s_finished['mean']:.2f} sec ({s_finished['mean']/60:.2f} min) "
              f"median={s_finished['median']:.2f} p90={s_finished['p90']:.2f} zeros={s_finished['zeros']}")
        print(f"  finished (excl teleports): count={s_finished_no_tele['count']} mean={s_finished_no_tele['mean']:.2f} sec ({s_finished_no_tele['mean']/60:.2f} min) "
              f"median={s_finished_no_tele['median']:.2f} p90={s_finished_no_tele['p90']:.2f} zeros={s_finished_no_tele['zeros']}")
        print(f"  teleported_count={len(teleport_ids)}")
        if tripinfo_stats:
            print(f"  SUMO tripinfo: count={tripinfo_stats['count']} mean_wait={tripinfo_stats['mean']:.2f} sec ({tripinfo_stats['mean']/60:.2f} min) "
                  f"median={tripinfo_stats['median']:.2f} p90={tripinfo_stats['p90']:.2f}")

        avg_wait_list.append(metric_to_use)

        traci.close()
        sys.stdout.flush()

    # --- Plotting (Same as dynamic model) ---
    xs = list(range(1, len(avg_wait_list) + 1))
    label = 'avg_wait_metric (static)'
    plt.figure(figsize=(8,5))
    plt.plot(xs, avg_wait_list, '-o', label=label)
    if len(avg_wait_list) >= 5:
        ma = np.convolve(avg_wait_list, np.ones(5)/5, mode='valid')
        plt.plot(list(range(5, 5+len(ma))), ma, linewidth=2, label='moving avg (k=5)')
    plt.xlabel("Run") # Changed from Epoch to Run for clarity
    plt.ylabel("Average waiting time metric (sec)")
    plt.title(f"Static Control curve ({model_name})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, f"avg_wait_vs_run_static_{model_name}.png")
    plt.savefig(plot_path)
    print("Saved plot to", plot_path)
    # plt.show() # Uncomment to show plot immediately

    mean_avg = np.mean(avg_wait_list)
    std_avg = np.std(avg_wait_list)
    print(f"STATIC SUMMARY: mean metric = {mean_avg:.2f} sec ({mean_avg/60:.2f} min), std = {std_avg:.2f} sec, runs = {len(avg_wait_list)}")


def get_options_static():
    # Simplified options for the static model
    parser = optparse.OptionParser()
    parser.add_option("-m", dest="model_name", type="string", default="static_model", help="name for the static run")
    parser.add_option("-e", dest="epochs", type="int", default=50, help="number of runs/epochs")
    parser.add_option("-s", dest="steps", type="int", default=500, help="simulation steps per run")
    parser.add_option("--test-cfg", dest="test_cfg", type="string", default=DEFAULT_TEST_CFG, help="sumo cfg file")
    parser.add_option("--nogui", action="store_true", dest="nogui", default=False, help="run headless sumo")
    parser.add_option("--seed", dest="seed", type="int", default=42, help="random seed")
    options, args = parser.parse_args()
    return options

if __name__ == "__main__":
    opts = get_options_static()
    model_name = opts.model_name
    epochs = opts.epochs
    steps = opts.steps
    test_cfg = opts.test_cfg
    nogui = opts.nogui
    seed = opts.seed

    run_static(model_name=model_name, epochs=epochs, steps=steps, cfg=test_cfg, nogui=nogui, seed=seed)