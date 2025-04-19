import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

# === Environment setup ===
env = gym.make("FrozenLake-v1", is_slippery=False, render_mode=None)  # Try False for easier mode

# === Q-table setup ===
q_table = np.zeros((env.observation_space.n, env.action_space.n))

# === Hyperparameters ===
alpha = 0.8              # Learning rate
gamma = 0.95             # Discount factor
epsilon = 1.0            # Initial exploration rate
epsilon_decay = 0.999
min_epsilon = 0.05
episodes = 15000         # Increased for slippery mode
max_steps = 100

# === Tracking metrics ===
episode_rewards = []
q_table_snapshots = []
snapshot_interval = 3000

# === Training loop ===
for episode in range(episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0

    for _ in range(max_steps):
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        q_table[state, action] = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)

        state = next_state
        total_reward += reward

        if done:
            break

    episode_rewards.append(total_reward)
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    if episode % snapshot_interval == 0:
        q_table_snapshots.append(q_table.copy())

print("Training finished. Final Q-table:")
print(np.round(q_table, 4))

# === Visualization: Q-table as heatmap ===
def plot_state_value_heatmap(q_table, grid_size=(4, 4), title="State Value Heatmap"):
    state_values = np.max(q_table, axis=1).reshape(grid_size)
    plt.figure(figsize=(6, 5))
    plt.imshow(state_values, cmap='viridis', interpolation='nearest')
    plt.title(title)
    plt.colorbar(label='Value')
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            value = state_values[i, j]
            color = 'white' if value < 0.5 else 'black'
            plt.text(j, i, f"{value:.2f}", ha='center', va='center', color=color)
    plt.xticks([]); plt.yticks([])
    plt.show()

plot_state_value_heatmap(q_table)

# === Visualization: reward per training episode ===
plt.figure(figsize=(8, 4))
plt.plot(episode_rewards)
plt.title("Training Reward per Episode")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid()
plt.show()

# === Optional: Q-table heatmap snapshots ===
for i, snapshot in enumerate(q_table_snapshots):
    plot_state_value_heatmap(snapshot, title=f"Q-Table at Episode {i * snapshot_interval}")

# === Evaluation: test the trained agent ===
test_episodes = 100
test_rewards = []
successes = 0

for episode in range(test_episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0

    for _ in range(max_steps):
        action = np.argmax(q_table[state])
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        if done:
            break

    test_rewards.append(total_reward)
    if total_reward > 0:
        successes += 1

print(f"\nTested {test_episodes} episodes")
print(f"Successes: {successes}")
print(f"Success rate: {successes / test_episodes * 100:.2f}%")
print(f"Average test reward: {np.mean(test_rewards):.4f}")

# === Visualization: test reward performance ===
plt.figure(figsize=(8, 4))
plt.plot(test_rewards)
plt.title("Test Rewards per Episode")
plt.xlabel("Test Episode")
plt.ylabel("Reward")
plt.grid()
plt.show()
