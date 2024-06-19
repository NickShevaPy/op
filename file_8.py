import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import numpy as np
from datetime import datetime

np.random.seed(10)
tf.compat.v1.set_random_seed(10)
# Так как это линейная регрессия, то y = W * X + b, где W и b – какие-то числа.
# Положим W = 0.5 и b = 1.4. Добавим также нормально распределенный случайный шум
W, b = 0.5, 1.4
#  создать набор данных, содержащий 100 примеров
X = np.linspace(0, 100, num=100)
# добавить случайный шум в метки y
y = np.random.normal(loc=W*X + b, scale=2.0, size=len(X))
# создать местрозаполнители
x_ph = tf.compat.v1.placeholder(shape=[None,], dtype=tf.float32)
y_ph = tf.compat.v1.placeholder(shape=[None,], dtype=tf.float32)
# создать переменные
v_weight = tf.compat.v1.get_variable("weight", shape=[1], dtype=tf.float32)
v_bias = tf.compat.v1.get_variable('bias', shape=[1], dtype=tf.float32)
# вычисление линейной функции
out = v_weight * x_ph + v_bias
# вычислить среднеквадратическую ошибку
loss = tf.reduce_mean((out-y_ph)**2)
# Теперь можно создать экземпляр оптимизатора и вызвать его метод minimize(),
# чтобы минимизировать СКО-потерю.
# Метод minimize() сначала вычисляет градиенты переменных (v_weight и v_bias),
# а затем с их помощью обновляет переменные
opt = tf.compat.v1.train.AdamOptimizer(0.4).minimize(loss)
# создаем сеанс
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
# цикл обучения параметров
for ep in range(210):
    # прогнать оптимизатор и получить потерю
    train_loss, _ = sess.run([loss, opt], feed_dict={x_ph:X, y_ph:y})
    # печатать номер эпохи и потерю
    if ep % 40 == 0:
        print(f'Эпоха: {ep}, СКО: {train_loss}, W: {sess.run(v_weight)}, b: {sess.run(v_bias)}')
        # В конце напечатаем окончательные значения переменных
print(f'Окончательный вес: {sess.run(v_weight)}, смещение: {sess.run(v_bias)}')


