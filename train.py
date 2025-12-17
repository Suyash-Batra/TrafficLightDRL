# train.py
# FULLY MODIFIED CODE: Includes fix for SUMO early termination (Simulation ended at time: -1.00)

from __future__ import absolute_import, print_function

import os
import sys
import optparse
import random
import numpy as np
import torch
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
from traci.exceptions import FatalTraCIError # Imported for safer step loop

DEFAULT_TRAIN_CFG = "pune.sumocfg"
DEFAULT_TEST_CFG = "pune_test.sumocfg"
TRIPINFO_OUTPUT = "maps/tripinfo.xml"
MODELS_DIR = "models"
PLOTS_DIR = "plots"
DEFAULT_MAX_VEHICLES_PER_LANE = 0.0

def ensure_dirs():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    maps_folder = os.path.dirname(TRIPINFO_OUTPUT)
    if maps_folder:
        os.makedirs(maps_folder, exist_ok=True)

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_vehicle_numbers(lanes):
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
    waiting_time = 0.0
    for lane in lanes:
        waiting_time += traci.lane.getWaitingTime(lane)
    return waiting_time

def phaseDuration(junction, phase_time, phase_state):
    traci.trafficlight.setRedYellowGreenState(junction, phase_state)
    traci.trafficlight.setPhaseDuration(junction, phase_time)

class Model(nn.Module):
    def __init__(self, input_dims, fc1_dims, fc2_dims, n_actions, lr):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(input_dims, fc1_dims)
        self.linear2 = nn.Linear(fc1_dims, fc2_dims)
        self.linear3 = nn.Linear(fc2_dims, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        actions = self.linear3(x)
        return actions

class Agent:
    def __init__(
        self,
        gamma,
        epsilon_start,
        lr,
        input_dims,
        fc1_dims,
        fc2_dims,
        batch_size,
        n_actions,
        junctions,
        max_memory_size=100000,
        epsilon_dec=1e-4,
        epsilon_end=0.05,
        replace_target=500,
    ):
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.lr = lr
        self.batch_size = batch_size
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.action_space = [i for i in range(n_actions)]
        self.junctions = junctions
        self.max_mem = max_memory_size
        self.epsilon_dec = epsilon_dec
        self.epsilon_end = epsilon_end
        self.mem_cntr = 0
        self.iter_cntr = 0
        self.replace_target = replace_target

        self.Q_eval = Model(self.input_dims, self.fc1_dims, self.fc2_dims, self.n_actions, lr=self.lr)
        self.Q_target = Model(self.input_dims, self.fc1_dims, self.fc2_dims, self.n_actions, lr=self.lr)
        self.Q_target.load_state_dict(self.Q_eval.state_dict())

        self.memory = dict()
        for junction in junctions:
            self.memory[junction] = {
                "state_memory": np.zeros((self.max_mem, self.input_dims), dtype=np.float32),
                "new_state_memory": np.zeros((self.max_mem, self.input_dims), dtype=np.float32),
                "reward_memory": np.zeros(self.max_mem, dtype=np.float32),
                "action_memory": np.zeros(self.max_mem, dtype=np.int32),
                "terminal_memory": np.zeros(self.max_mem, dtype=np.bool_),
                "mem_cntr": 0,
            }

    def store_transition(self, state, state_, action, reward, done, junction):
        s = np.array(state, dtype=np.float32).reshape(-1)[: self.input_dims]
        s_ = np.array(state_, dtype=np.float32).reshape(-1)[: self.input_dims]
        if s.shape[0] < self.input_dims:
            s = np.pad(s, (0, self.input_dims - s.shape[0]), 'constant', constant_values=0.0)
        if s_.shape[0] < self.input_dims:
            s_ = np.pad(s_, (0, self.input_dims - s_.shape[0]), 'constant', constant_values=0.0)

        index = self.memory[junction]["mem_cntr"] % self.max_mem
        self.memory[junction]["state_memory"][index] = s
        self.memory[junction]["new_state_memory"][index] = s_
        self.memory[junction]["reward_memory"][index] = reward
        self.memory[junction]["terminal_memory"][index] = done
        self.memory[junction]["action_memory"][index] = action
        self.memory[junction]["mem_cntr"] += 1
        self.mem_cntr += 1

    def choose_action(self, observation):
        if not isinstance(observation, (list, np.ndarray)):
            observation = [observation]
        obs = np.array(observation, dtype=np.float32).reshape(-1)[: self.input_dims]
        if obs.shape[0] < self.input_dims:
            obs = np.pad(obs, (0, self.input_dims - obs.shape[0]), 'constant', constant_values=0.0)
        state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.Q_eval.device)

        if np.random.random() > self.epsilon:
            with torch.no_grad():
                actions = self.Q_eval.forward(state)
                action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return int(action)

    def reset_memory_counters(self):
        for j in self.memory:
            self.memory[j]["mem_cntr"] = 0

    def save(self, model_name):
        torch.save(self.Q_eval.state_dict(), os.path.join(MODELS_DIR, f"{model_name}.bin"))

    def load(self, model_name):
        path = os.path.join(MODELS_DIR, f"{model_name}.bin")
        if os.path.exists(path):
            self.Q_eval.load_state_dict(torch.load(path, map_location=self.Q_eval.device))
            self.Q_target.load_state_dict(self.Q_eval.state_dict())
            print("Loaded model from", path)
            return True
        return False

    def learn(self, junction):
        mem = self.memory[junction]
        mem_size = min(mem["mem_cntr"], self.max_mem)
        if mem_size < self.batch_size:
            return

        batch = np.random.choice(mem_size, self.batch_size, replace=False)

        state_batch = torch.tensor(mem["state_memory"][batch]).to(self.Q_eval.device)
        new_state_batch = torch.tensor(mem["new_state_memory"][batch]).to(self.Q_eval.device)
        reward_batch = torch.tensor(mem["reward_memory"][batch]).to(self.Q_eval.device)
        terminal_batch = torch.tensor(mem["terminal_memory"][batch]).to(self.Q_eval.device)
        action_batch = torch.tensor(mem["action_memory"][batch]).long().to(self.Q_eval.device)

        q_eval_all = self.Q_eval.forward(state_batch)
        q_eval = q_eval_all.gather(1, action_batch.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            q_next_online = self.Q_eval.forward(new_state_batch)
            best_actions = torch.argmax(q_next_online, dim=1)
            q_next_target_all = self.Q_target.forward(new_state_batch)
            q_next = q_next_target_all.gather(1, best_actions.unsqueeze(1)).squeeze(1)
            q_next[terminal_batch] = 0.0
            q_target = reward_batch + self.gamma * q_next

        loss = F.mse_loss(q_eval, q_target)
        self.Q_eval.optimizer.zero_grad()
        loss.backward()
        self.Q_eval.optimizer.step()

        self.iter_cntr += 1
        if self.epsilon > self.epsilon_end:
            self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_dec)
        if self.iter_cntr % self.replace_target == 0:
            self.Q_target.load_state_dict(self.Q_eval.state_dict())

def detect_max_input_dims(cfg_list, gui=False):
    max_dim = 1
    for cfg in cfg_list:
        if not cfg or not os.path.exists(cfg):
            continue
        try:
            # We use checkBinary("sumo") if nogui is requested for consistency
            bin = checkBinary("sumo") if not gui else checkBinary("sumo-gui") 
            traci.start([bin, "-c", cfg])
            juncs = traci.trafficlight.getIDList()
            if len(juncs) == 0:
                traci.close()
                continue
            lanes = traci.trafficlight.getControlledLanes(juncs[0])
            dim = len(lanes) if lanes is not None else 1
            max_dim = max(max_dim, dim)
            traci.close()
        except Exception as e:
            try:
                traci.close()
            except:
                pass
            print("Warning: failed warmup for cfg", cfg, ":", e)
    return max_dim

def run(train=True, model_name="model", epochs=50, steps=500, train_cfgs=None, test_cfg=None,
        max_veh_per_lane=DEFAULT_MAX_VEHICLES_PER_LANE, nogui=False, seed=42, fine_tune_epochs=0):
    ensure_dirs()
    set_seeds(seed)

    train_cfg_list = []
    if train_cfgs:
        train_cfg_list = [c.strip() for c in train_cfgs.split(",") if c.strip()]
    if len(train_cfg_list) == 0 and train:
        train_cfg_list = [DEFAULT_TRAIN_CFG]

    test_cfg = test_cfg or DEFAULT_TEST_CFG

    cfgs_to_probe = train_cfg_list + ([test_cfg] if test_cfg else [])
    print("Detecting input dims across configs:", cfgs_to_probe)
    # Pass nogui to detect_max_input_dims to choose the correct binary
    input_dims = detect_max_input_dims(cfgs_to_probe, gui=(not nogui))
    print("Using input dims (max over configs):", input_dims)

    gamma = 0.99
    epsilon_start = 1.0
    epsilon_end = 0.02
    total_estimated_learn_calls = max(1, epochs * max(1, int(steps/10)))
    epsilon_dec = (epsilon_start - epsilon_end) / (total_estimated_learn_calls)
    lr = 1e-4
    fc1 = 128
    fc2 = 64
    batch_size = 64
    n_actions = 4
    replace_target = 500

    if max_veh_per_lane <= 0:
        MAX_VEHICLES_PER_LANE = 50.0
    else:
        MAX_VEHICLES_PER_LANE = max_veh_per_lane

    dummy_junctions = [0]
    brain = Agent(
        gamma=gamma,
        epsilon_start=epsilon_start,
        lr=lr,
        input_dims=input_dims,
        fc1_dims=fc1,
        fc2_dims=fc2,
        batch_size=batch_size,
        n_actions=n_actions,
        junctions=dummy_junctions,
        max_memory_size=100000,
        epsilon_dec=epsilon_dec,
        epsilon_end=epsilon_end,
        replace_target=replace_target,
    )

    if os.path.exists(os.path.join(MODELS_DIR, f"{model_name}.bin")):
        print("Found existing model. Loading weights (if you want fresh training, remove or rename the bin file).")
        brain.load(model_name)

    best_metric = np.inf
    avg_wait_list = []

    for e in range(epochs):
        if train:
            cfg = random.choice(train_cfg_list) if len(train_cfg_list) > 1 else train_cfg_list[0]
        else:
            cfg = test_cfg

        sumo_bin = checkBinary("sumo") if nogui else checkBinary("sumo-gui")
        seed_arg = ["--seed", str(seed + e)]
        
        # --- CRITICAL FIX: Add --end flag to prevent early termination ---
        sumo_cmd = [
            sumo_bin, 
            "-c", cfg, 
            "--tripinfo-output", TRIPINFO_OUTPUT,
            "--end", str(steps) # Forces SUMO to run until time 'steps'
        ] + seed_arg
        
        try:
            traci.start(sumo_cmd)
        except Exception:
            try:
                traci.close()
            except:
                pass
            # Fallback if the first attempt fails
            traci.start([sumo_bin, "-c", cfg, "--tripinfo-output", TRIPINFO_OUTPUT, "--end", str(steps)]) 
        # -----------------------------------------------------------------

        all_junctions = traci.trafficlight.getIDList()
        if len(all_junctions) == 0:
            print("Warning: no traffic lights in cfg", cfg)
            traci.close()
            continue
        junction_numbers = list(range(len(all_junctions)))

        for jn_idx in junction_numbers:
            if jn_idx not in brain.memory:
                brain.memory[jn_idx] = {
                    "state_memory": np.zeros((brain.max_mem, brain.input_dims), dtype=np.float32),
                    "new_state_memory": np.zeros((brain.max_mem, brain.input_dims), dtype=np.float32),
                    "reward_memory": np.zeros(brain.max_mem, dtype=np.float32),
                    "action_memory": np.zeros(brain.max_mem, dtype=np.int32),
                    "terminal_memory": np.zeros(brain.max_mem, dtype=np.bool_),
                    "mem_cntr": 0,
                }

        if e == 0:
            max_lanes = 0
            for j in all_junctions:
                lanes = traci.trafficlight.getControlledLanes(j)
                max_lanes = max(max_lanes, len(lanes))
            if max_lanes > brain.input_dims:
                print(f"Updating model input dims from {brain.input_dims} to {max_lanes}")
                brain.input_dims = max_lanes
                brain = Agent(
                    gamma=gamma,
                    epsilon_start=brain.epsilon,
                    lr=lr,
                    input_dims=brain.input_dims,
                    fc1_dims=fc1,
                    fc2_dims=fc2,
                    batch_size=batch_size,
                    n_actions=n_actions,
                    junctions=junction_numbers,
                    max_memory_size=100000,
                    epsilon_dec=epsilon_dec,
                    epsilon_end=epsilon_end,
                    replace_target=replace_target,
                )
                if os.path.exists(os.path.join(MODELS_DIR, f"{model_name}.bin")):
                    brain.load(model_name)

            if max_veh_per_lane <= 0:
                sample_counts = []
                for _ in range(5):
                    traci.simulationStep()
                    for j in all_junctions:
                        lanes = traci.trafficlight.getControlledLanes(j)
                        sample_counts.extend(get_vehicle_numbers(lanes))
                avg = max(1.0, np.mean(sample_counts) * 2.0)
                MAX_VEHICLES_PER_LANE = max(20.0, float(avg))
                print("Auto-set MAX_VEHICLES_PER_LANE to", MAX_VEHICLES_PER_LANE)

        prev_vehicles_per_lane = {i: [0] * brain.input_dims for i in junction_numbers}
        prev_wait_time = {j: 0.0 for j in all_junctions}
        prev_action = {i: 0 for i in junction_numbers}
        traffic_lights_time = {j: 0 for j in all_junctions}

        veh_waits = {}    # vid -> last seen accumulated waiting time (sec)
        arrived = set()   # finished vehicle ids
        seen_vehicles = set()
        teleport_ids = set()

        select_lane = [
            ["yyyrrrrrrrrr", "GGGrrrrrrrrr"],
            ["rrryyyrrrrrr", "rrrGGGrrrrrr"],
            ["rrrrrryyyrrr", "rrrrrrGGGrrr"],
            ["rrrrrrrrryyy", "rrrrrrrrrGGG"],
        ]

        step = 0
        min_duration = 15

        if not train:
            brain.epsilon = 0.0

        while step <= steps:
            try:
                traci.simulationStep()
            except FatalTraCIError:
                # Catch the error if the simulation already ended (even with --end)
                print(f"Warning: SUMO ended prematurely at step {step}. Breaking loop.")
                break

            # update current vehicles' accumulated waiting time
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

            # collect arrived vehicle ids (they finished trips this step)
            try:
                arrived_ids = traci.simulation.getArrivedIDList()
                if arrived_ids:
                    arrived.update(arrived_ids)
            except Exception:
                pass

            # detect teleports this step
            try:
                started_tele = traci.simulation.getStartingTeleportIDList()
                ended_tele = traci.simulation.getEndingTeleportIDList()
                if started_tele:
                    teleport_ids.update(started_tele)
                if ended_tele:
                    teleport_ids.update(ended_tele)
            except Exception:
                pass

            for jn_idx, junction in enumerate(all_junctions):
                controlled_lanes = traci.trafficlight.getControlledLanes(junction)

                # lane waiting used only for reward shaping
                waiting_time = get_waiting_time(controlled_lanes)

                if traffic_lights_time[junction] == 0:
                    vehicles_per_lane = get_vehicle_numbers(controlled_lanes)
                    state_np = np.array(vehicles_per_lane, dtype=np.float32)
                    if state_np.shape[0] < brain.input_dims:
                        state_np = np.pad(state_np, (0, brain.input_dims - state_np.shape[0]), 'constant')
                    elif state_np.shape[0] > brain.input_dims:
                        state_np = state_np[:brain.input_dims]

                    state_norm = (state_np / MAX_VEHICLES_PER_LANE).tolist()

                    prev_state_np = np.array(prev_vehicles_per_lane[jn_idx], dtype=np.float32)
                    prev_state_norm = (prev_state_np / MAX_VEHICLES_PER_LANE).tolist()

                    prev_vehicles_per_lane[jn_idx] = state_np.tolist()

                    reward = (prev_wait_time[junction] - waiting_time) - 0.01 * (waiting_time)
                    prev_wait_time[junction] = waiting_time

                    done = (step == steps)
                    brain.store_transition(prev_state_norm, state_norm, prev_action[jn_idx], reward, done, jn_idx)

                    lane_action = brain.choose_action(state_norm)
                    prev_action[jn_idx] = lane_action
                    phaseDuration(junction, 6, select_lane[lane_action][0])
                    phaseDuration(junction, min_duration + 10, select_lane[lane_action][1])

                    traffic_lights_time[junction] = min_duration + 10

                    if train:
                        brain.learn(jn_idx)
                else:
                    traffic_lights_time[junction] -= 1
            step += 1

        # --- diagnostics & robust averages ---
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

        # decide metric for model selection
        metric_to_use = s_all["mean"]
        

        # print diagnostics
        print(f"{'TRAIN' if train else 'TEST'} epoch {e+1}/{epochs} (cfg={cfg}) epsilon={brain.epsilon:.3f}")
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
        if train and metric_to_use < best_metric:
            best_metric = metric_to_use
            print("New best model found - saving.")
            brain.save(model_name)

        traci.close()
        sys.stdout.flush()

        if (not train) and fine_tune_epochs > 0:
            print("Fine-tuning on test environment for", fine_tune_epochs, "epochs...")
            brain.epsilon = max(brain.epsilon, 0.1)
            for ft in range(fine_tune_epochs):
                traci.start([sumo_bin, "-c", cfg])
                all_junctions_ft = traci.trafficlight.getIDList()
                step_ft = 0
                while step_ft <= int(steps / 2):
                    traci.simulationStep()
                    for jn_idx, junction in enumerate(all_junctions_ft):
                        controlled_lanes = traci.trafficlight.getControlledLanes(junction)
                        waiting_time = get_waiting_time(controlled_lanes)
                        vehicles_per_lane = get_vehicle_numbers(controlled_lanes)
                        state_np = np.array(vehicles_per_lane, dtype=np.float32)
                        if state_np.shape[0] < brain.input_dims:
                            state_np = np.pad(state_np, (0, brain.input_dims - state_np.shape[0]), 'constant')
                        state_norm = (state_np / MAX_VEHICLES_PER_LANE).tolist()
                        action = brain.choose_action(state_norm)
                        phaseDuration(junction, 6, select_lane[action][0])
                        phaseDuration(junction, min_duration + 10, select_lane[action][1])
                    step_ft += 1
                traci.close()
            print("Fine-tuning done. Returning to test mode.")

    xs = list(range(1, len(avg_wait_list) + 1))
    label = 'avg_wait_metric (train)' if train else 'avg_wait_metric (test)'
    plt.figure(figsize=(8,5))
    plt.plot(xs, avg_wait_list, '-o', label=label)
    if len(avg_wait_list) >= 5:
        ma = np.convolve(avg_wait_list, np.ones(5)/5, mode='valid')
        plt.plot(list(range(5, 5+len(ma))), ma, linewidth=2, label='moving avg (k=5)')
    plt.xlabel("Epoch")
    plt.ylabel("Average waiting time metric (sec)")
    plt.title(f"{'Training' if train else 'Testing'} curve ({model_name})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    suffix = "train" if train else "test"
    plot_path = os.path.join(PLOTS_DIR, f"avg_wait_vs_epoch_{suffix}_{model_name}.png")
    plt.savefig(plot_path)
    # plt.show() # Uncomment to show plot immediately

    if not train:
        mean_avg = np.mean(avg_wait_list)
        std_avg = np.std(avg_wait_list)
        print(f"TEST SUMMARY: mean metric = {mean_avg:.2f} sec ({mean_avg/60:.2f} min), std = {std_avg:.2f} sec, runs = {len(avg_wait_list)}")

def get_options():
    parser = optparse.OptionParser()
    parser.add_option("-m", dest="model_name", type="string", default="model", help="model name")
    parser.add_option("--train", action="store_true", dest="train", default=False, help="training flag")
    parser.add_option("-e", dest="epochs", type="int", default=50, help="number of epochs")
    parser.add_option("-s", dest="steps", type="int", default=500, help="simulation steps per epoch")
    parser.add_option("--train-cfgs", dest="train_cfgs", type="string", default="", help="comma-separated train sumo cfg files (domain randomization)")
    parser.add_option("--train-cfg", dest="train_cfg", type="string", default=DEFAULT_TRAIN_CFG, help="single train sumo cfg")
    parser.add_option("--test-cfg", dest="test_cfg", type="string", default=DEFAULT_TEST_CFG, help="test sumo cfg")
    parser.add_option("--max-veh-per-lane", dest="max_veh_per_lane", type="float", default=DEFAULT_MAX_VEHICLES_PER_LANE, help="normalization constant (0 to auto-estimate)")
    parser.add_option("--nogui", action="store_true", dest="nogui", default=False, help="run headless sumo")
    parser.add_option("--seed", dest="seed", type="int", default=42, help="random seed")
    parser.add_option("--fine-tune", dest="fine_tune_epochs", type="int", default=0, help="fine-tune epochs on test cfg (0 = disabled)")
    options, args = parser.parse_args()
    if not options.train_cfgs and options.train_cfg:
        options.train_cfgs = options.train_cfg
    return options

if __name__ == "__main__":
    opts = get_options()
    model_name = opts.model_name
    train_flag = opts.train
    epochs = opts.epochs
    steps = opts.steps
    train_cfgs = opts.train_cfgs
    test_cfg = opts.test_cfg
    max_veh = opts.max_veh_per_lane
    nogui = opts.nogui
    seed = opts.seed
    fine_tune_epochs = opts.fine_tune_epochs

    run(train=train_flag, model_name=model_name, epochs=epochs, steps=steps,
        train_cfgs=train_cfgs, test_cfg=test_cfg, max_veh_per_lane=max_veh,
        nogui=nogui, seed=seed, fine_tune_epochs=fine_tune_epochs)