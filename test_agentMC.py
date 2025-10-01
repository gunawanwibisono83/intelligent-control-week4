import gymnasium as gym
import numpy as np
from tensorflow.keras.models import load_model

# Buat environment dengan render mode human
env = gym.make('MountainCar-v0', render_mode="human")
state_size = env.observation_space.shape[0]

# Load model hasil training
model = load_model(r"c:\Users\ASUS\Documents\DOKUMEN GUN\KULIAH\SEMESTER 7\2. MATKUL\8. PRAKTIKUM KONTROL CERDAS\Tugas\M4\Assigment\dqn_mountaincar.h5", compile=False)
model.compile(loss="mse", optimizer="adam")

# Jalankan beberapa episode untuk uji coba
for e in range(25):
    state, _ = env.reset()
    state = np.reshape(state, [1, state_size])
    total_reward = 0
    for time in range(20):  # max step MountainCar
        q_values = model.predict(state, verbose=0)
        action = np.argmax(q_values[0])
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = np.reshape(next_state, [1, state_size])
        total_reward += reward
        if done:
            print(f"Test Episode: {e+1}, Total Reward: {total_reward:.2f}")
            break

env.close()
