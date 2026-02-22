import numpy as np
import torch
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle

from agents import BaselineLinUCB, PretrainedLinUCB
from models import InferenceModel

# Main entry for online evaluation.
if __name__ == "__main__":
    # Parameter configuration.
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    DATASET = os.getenv("NCB_DATASET", "taobao")
    BASE_DATA_PATH = os.getenv("NCB_BASE_DATA_PATH", os.path.join(ROOT_DIR, "data", DATASET, "preprocess"))
    MODEL_LOAD_PATH = os.path.join(os.path.dirname(__file__), "pretrain_models")
    OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs", "real")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    K = 20
    ALPHA = 1.0
    SIMULATION_SEEDS = [123, 456, 789, 101, 202]
    
    # Load data and model once.
    print("--- Loading Data and Models ---")
    Y_online = np.load(os.path.join(BASE_DATA_PATH, f"Y_online_{DATASET}.npy"))
    R_online = np.load(os.path.join(BASE_DATA_PATH, f"R_online_{DATASET}.npy"))
    num_online_samples = Y_online.shape[0]
    print(f"Online evaluation set size: {num_online_samples} samples")

    FULL_DIMS = Y_online.shape[1]
    OBSERVED_DIMS = 16
    
    inference_model = InferenceModel(input_dim=OBSERVED_DIMS, output_dim=FULL_DIMS - OBSERVED_DIMS, hidden_dim=128)
    inference_model.load_state_dict(torch.load(os.path.join(MODEL_LOAD_PATH, f'MLP.pth')))
    
    num_steps = Y_online.shape[0] - K

    all_runs_results = {
        "Pretrained LinUCB": [],
        "Oracle LinUCB": [],
        "Misspecified LinUCB": []
    }

    for seed in SIMULATION_SEEDS:
        print(f"\n===== Running Simulation with Seed: {seed} =====")
        # Initialize and run each agent.
        agents_to_run = {
            "Pretrained LinUCB": PretrainedLinUCB(feature_dim=FULL_DIMS, observed_dim=OBSERVED_DIMS, inference_model=inference_model, alpha=ALPHA),
            "Oracle LinUCB": BaselineLinUCB(feature_dim=FULL_DIMS, alpha=ALPHA),
            "Misspecified LinUCB": BaselineLinUCB(feature_dim=OBSERVED_DIMS, alpha=ALPHA)
        }

        for agent_name, agent in agents_to_run.items():
            print(f"\n--- Running Simulation for: {agent_name} ---")
            np.random.seed(seed)
            total_reward = 0
            rewards_history = []
            ctr_history = []
            
            for t in tqdm(range(num_steps), desc=agent_name):
                candidate_indices = np.random.choice(Y_online.shape[0], K, replace=False)
                
                if agent_name == "Pretrained LinUCB":
                    contexts = Y_online[candidate_indices, :OBSERVED_DIMS]
                    chosen_local_index = agent.select_arm(contexts)
                elif agent_name == "Oracle LinUCB":
                    contexts = Y_online[candidate_indices] 
                    chosen_local_index = agent.select_arm(contexts)
                elif agent_name == "Misspecified LinUCB":
                    contexts = Y_online[candidate_indices, :OBSERVED_DIMS] 
                    chosen_local_index = agent.select_arm(contexts)
                else:
                    raise ValueError(f"Unexpected agent name: {agent_name}")

                chosen_global_index = candidate_indices[chosen_local_index]
                reward = R_online[chosen_global_index]
                
                if agent_name == "Pretrained LinUCB":
                    chosen_context = Y_online[chosen_global_index, :OBSERVED_DIMS]
                    agent.update(chosen_context, reward)
                elif agent_name == "Oracle LinUCB":
                    chosen_context = Y_online[chosen_global_index]
                    agent.update(chosen_context, reward)
                elif agent_name == "Misspecified LinUCB":
                    chosen_context = Y_online[chosen_global_index, :OBSERVED_DIMS]
                    agent.update(chosen_context, reward)

                total_reward += reward
                rewards_history.append(reward)
                ctr_history.append(total_reward / (t + 1))
            
            all_runs_results[agent_name].append(ctr_history)
            print(f"Final CTR for {agent_name}: {ctr_history[-1]:.4%}")
            print(f"Overall Click-Through Rate (CTR): {total_reward / (num_online_samples - K):.4%}")
            print(f"Total reward obtained: {total_reward}")

    with open(os.path.join(OUTPUT_DIR, "all_runs_results.pkl"), 'wb') as f:
        pickle.dump(all_runs_results, f)

    # Visualize and compare results.
    print("\n--- Plotting Comparison ---")
    plt.figure(figsize=(12, 8))
    
    colors = {
        "Pretrained LinUCB": "blue",
        "Oracle LinUCB": "green",
        "Misspecified LinUCB": "red"
    }
    
    for agent_name, histories in all_runs_results.items():
        histories_np = np.array(histories)
        mean_ctr = np.mean(histories_np, axis=0)
        std_ctr = np.std(histories_np, axis=0)
        
        # Compute standard error.
        stderr_ctr = std_ctr / np.sqrt(len(SIMULATION_SEEDS))
        
        # Plot mean curve.
        plt.plot(mean_ctr, label=agent_name, color=colors[agent_name], linewidth=2)
        
        plt.fill_between(
            range(num_steps),
            mean_ctr - 1.96 * stderr_ctr,
            mean_ctr + 1.96 * stderr_ctr,
            color=colors[agent_name],
            alpha=0.2
        )

    plt.title(f'Algorithm Comparison on {DATASET} (Avg. over {len(SIMULATION_SEEDS)} runs)', fontsize=16)
    plt.xlabel('Time Step (t)', fontsize=12)
    plt.ylabel('Cumulative Click-Through Rate (CTR)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.ylim(0, 0.10) 
    plt.savefig(os.path.join(OUTPUT_DIR, "final_comparison_plot.png"))
    plt.show()
