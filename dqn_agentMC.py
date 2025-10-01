import gymnasium as gym
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=5000)   # buffer diperbesar
        self.gamma = 0.95                  # lebih fokus ke reward dekat
        self.epsilon = 1.0                 # mulai full eksplorasi
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999         # eksplorasi diperlambat
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Input(shape=(self.state_size,)))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    episodes = 1000   # kalau kuat device bisa ditambah
    batch_size = 64

    scores = []

    for e in range(episodes):
        state, _ = env.reset()
        state = np.reshape(state, [1, state_size])
        total_reward = 0
        for time in range(200):  # max step MountainCar
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # ✅ Reward shaping
            position = next_state[0]
            reward = reward + abs(position + 0.5)  # makin kanan makin besar

            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                print(f"Episode: {e+1}/{episodes}, Score: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
                break
        scores.append(total_reward)
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

    # ✅ Simpan model
    agent.model.save("dqn_mountaincar.h5")
    print("Model saved to dqn_mountaincar.h5")

    env.close()

    # ✅ Plot grafik skor
    plt.plot(scores)
    plt.xlabel('Episode')
    plt.ylabel('Score (Total Reward)')
    plt.title('Training Progress on MountainCar-v0 with Reward Shaping')
    plt.show()
