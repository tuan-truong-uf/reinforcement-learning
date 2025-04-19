```markdown
# 🧠 Reinforcement Learning Q-Agents

This repository contains classical Q-learning agents implemented from scratch using `gymnasium`. The agents are trained to solve two fundamental RL environments: `FrozenLake-v1` and `Taxi-v3`. The project includes reward tracking, value visualization, live testing with rendering, and analysis of training performance.

## 🚀 Highlights
- ✅ Q-learning algorithm with ε-greedy exploration
- ✅ Visualizations of agent performance (reward curves, heatmaps)
- ✅ Real-time rendering of the trained agents
- ✅ Class-based modular structure for reuse and testing
- ✅ Support for noisy environments (`is_slippery=True`)

## 🧠 Environments
| Environment | Description |
|-------------|-------------|
| `FrozenLake-v1` | Navigate a slippery 4x4 grid without falling into holes |
| `Taxi-v3`       | Pick up and drop off passengers in a 5x5 grid world with a 500-state space |

## 🧪 How to Run

### 🚥 Install dependencies
```bash
pip install -r requirements.txt
```

### 🧊 FrozenLake Agent (Deterministic)
```bash
python frozen_lake.py
```

### 🚕 Taxi Agent (with rendering)
```bash
python taxi.py
```

## 📊 Visualizations
- 📈 Reward per episode tracking
- 🟨 Heatmaps of state values from the Q-table
- 🎮 Live rendering of the agent’s behavior

## 💡 Future Work
- Add SARSA and Dyna-Q variants for comparison  
- Extend to continuous environments using function approximation  
- Log Q-table evolution with TensorBoard or Weights & Biases



---
