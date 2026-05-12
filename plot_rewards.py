import matplotlib.pyplot as plt
import pandas as pd

def extract_summary_data(file_path):
    episodes, rewards = [], []
    print(f"Processing {file_path}...")
    
    with open(file_path, 'r') as file:
        for line in file:
            if "SUMMARY for ep" in line:
                try:
                    parts = line.split(':')
                    ep_part = parts[1].strip()
                    episode_num = int(ep_part.split(' ')[-1])
                    reward_part = parts[2].split('=')[-1].strip()
                    reward_val = float(reward_part)
                    
                    episodes.append(episode_num)
                    rewards.append(reward_val)
                except (ValueError, IndexError):
                    continue
    return episodes, rewards

def plot_rewards(episodes, rewards):
    plt.figure(figsize=(14, 7))
    
    # Calculate Moving Average
    series = pd.Series(rewards)
    rolling_avg = series.rolling(window=100).mean()
    
    # 1. Plot Raw Data
    plt.plot(episodes, rewards, color='#2ca02c', alpha=0.15, label='Total Reward')
    
    # 2. Plot Moving Average
    plt.plot(episodes, rolling_avg, color='#d62728', linewidth=1.5, label='100-Ep Moving Avg')

    # --- Vertical Lines (Adjusted to 20k and 40k) ---
    plt.axvline(x=20000, color='blue', linestyle='--', alpha=0.7, label='End P1 Training (20k)')
    plt.axvline(x=40000, color='purple', linestyle='--', alpha=0.7, label='End P2 Training (40k)')

    # --- Zero Crossing Check (Only after Episode 40,000) ---
    zero_crossing_ep = None
    for i, val in enumerate(rolling_avg):
        # Check if current episode is > 40000 and avg reward >= 0
        if episodes[i] > 50000 and val <= 0:
            zero_crossing_ep = episodes[i]
            break
    
    if zero_crossing_ep is not None:
        plt.axvline(x=zero_crossing_ep, color='black', linestyle=':', linewidth=2, 
                    label=f'Avg Reward 0 @ Ep {zero_crossing_ep}')
        print(f"Moving average reached 0 at episode: {zero_crossing_ep}")
    else:
        print("Moving average did not reach 0 after episode 40,000.")

    # Formatting
    plt.title('Agent Training Progress (P1 & P2 Phases)')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    output_path = "training_plot_final.png"
    plt.savefig(output_path, dpi=300)
    print(f"Successfully saved plot to {output_path}")

# --- Execution ---
FILE_NAME = "./training results/lidar without view/logs/player_one_log.csv"
eps, rews = extract_summary_data(FILE_NAME)

if eps:
    plot_rewards(eps, rews)
else:
    print("No summary data found.")