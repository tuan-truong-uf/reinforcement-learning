import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym


class QLearningAgent:
    def __init__(self, env, alpha=0.7, gamma=0.95, epsilon=1.0,
                 epsilon_decay=0.9995, min_epsilon=0.1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.q_table = np.zeros((env.observation_space.n, env.action_space.n))
        self.rewards = []
        self.successes = 0
        self.snapshots = []

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.q_table[state])

    def update_q(self, state, action, reward, next_state):
        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state])
        self.q_table[state, action] = (1 - self.alpha) * old_value + \
                                      self.alpha * (reward + self.gamma * next_max)

    def train(self, episodes=30000, max_steps=200, snapshot_interval=5000):
        for episode in range(1, episodes + 1):
            state, _ = self.env.reset()
            done = False
            total_reward = 0

            for _ in range(max_steps):
                action = self.choose_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                self.update_q(state, action, reward, next_state)
                total_reward += reward
                state = next_state
                if terminated or truncated:
                    if reward == 20:
                        self.successes += 1
                    break

            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            self.rewards.append(total_reward)

            if episode % 1000 == 0:
                print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {self.epsilon:.4f}")

            if episode % snapshot_interval == 0:
                self.snapshots.append(self.q_table.copy())

    def test(self, episodes=100, max_steps=200, render=False):
        success_count = 0
        test_rewards = []

        for _ in range(episodes):
            state, _ = self.env.reset()
            total_reward = 0

            for _ in range(max_steps):
                if render:
                    self.env.render()
                action = np.argmax(self.q_table[state])
                state, reward, terminated, truncated, _ = self.env.step(action)
                total_reward += reward
                if terminated or truncated:
                    break

            test_rewards.append(total_reward)
            if total_reward >= 20:
                success_count += 1

        print(f"\nTested {episodes} episodes")
        print(f"Successes: {success_count}")
        print(f"Success rate: {success_count / episodes * 100:.2f}%")
        print(f"Average reward: {np.mean(test_rewards):.2f}")

        plt.figure(figsize=(8, 4))
        plt.plot(test_rewards)
        plt.title("Test Rewards per Episode")
        plt.xlabel("Test Episode")
        plt.ylabel("Reward")
        plt.grid()
        plt.show()

    def plot_training_rewards(self):
        plt.figure(figsize=(8, 4))
        plt.plot(self.rewards)
        plt.title("Training Reward per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.grid()
        plt.show()

    def plot_q_snapshots(self):
        for i, snapshot in enumerate(self.snapshots):
            self.plot_q_heatmap(snapshot, title=f"Q-table at episode {(i + 1) * 5000}")

    def plot_q_heatmap(self, q_table=None, title="Q-table Heatmap"):
        if q_table is None:
            q_table = self.q_table
        state_values = np.max(q_table, axis=1).reshape(25, -1)[:5, :5]  # visualize a 5x5 block
        plt.figure(figsize=(6, 5))
        plt.imshow(state_values, cmap='viridis', interpolation='nearest')
        plt.title(title)
        plt.colorbar(label='Value')
        for i in range(state_values.shape[0]):
            for j in range(state_values.shape[1]):
                value = state_values[i, j]
                color = 'white' if value < 0.5 else 'black'
                plt.text(j, i, f"{value:.2f}", ha='center', va='center', color=color)
        plt.xticks([])
        plt.yticks([])
        plt.show()


# === Run Everything ===
env = gym.make("Taxi-v3", render_mode="human")
agent = QLearningAgent(env)

print("\n--- Training the agent ---")
agent.train()

print("\n--- Visualizing Training Results ---")
agent.plot_training_rewards()
agent.plot_q_snapshots()

print("\n--- Testing the agent ---")
agent.test(render=True)

env.close()
