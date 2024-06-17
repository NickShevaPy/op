import gymnasium as gym
import matplotlib.pyplot as plt

# Создание среды CartPole-v1
env = gym.make("CartPole-v1", render_mode='rgb_array')
obs = env.reset()
for _ in range(10):
    # Получение кадра рендеринга
    img = env.render()

    # Отображение кадра с использованием Matplotlib
    plt.imshow(img)
    plt.show()

    # Случайное действие
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()