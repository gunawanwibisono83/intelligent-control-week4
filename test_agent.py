import gymnasium as gym
import numpy as np
from dqn_agent import DQNAgent
from tensorflow.keras.models import load_model

# Buat environment dengan render mode human
env = gym.make('CartPole-v1', render_mode="human")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Inisialisasi agen
agent = DQNAgent(state_size, action_size)

# Load model hasil training (pakai compile=False agar tidak error di TensorFlow baru)
agent.model = load_model("dqn_cartpole.h5", compile=False)

# Re-compile model manual (pakai loss dan optimizer yang sama seperti waktu training)
agent.model.compile(loss="mse", optimizer="adam")

# Minimalkan eksplorasi saat testing
agent.epsilon = 0.01

# Jalankan beberapa episode untuk uji coba
for e in range(25):
    state, _ = env.reset()
    state = np.reshape(state, [1, state_size])
    for time in range(500):
        action = agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = np.reshape(next_state, [1, state_size])
        if done:
            print(f"Test Episode: {e+1}, Score: {time}")
            break

env.close()
