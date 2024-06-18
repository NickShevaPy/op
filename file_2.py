import gymnasium as gym

# Создание и инициализировать окружающую среду
env = gym.make('CartPole-v1')
env.reset()
# сыграть 10 игр
for i in range(10):
    # инициализировать переменные
    done = False
    game_rew = 0
    while not done:
        # выбрать случайное действие
        act = env.action_space.sample()
        # выполнить один шаг взаимодействия с окружающей средой
        obs, rew, done, _, _ = env.step(act)
        game_rew += rew
        # если завершено, напечатать полное вознаграждение в игреи сбросить среду
        if done:
            print(f"Эпизод {i} заверщен, Вознаграждение: {game_rew}")
            env.reset()
