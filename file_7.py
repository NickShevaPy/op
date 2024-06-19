import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import numpy as np

"""Переменная – это изменяемый тензор, который можно обучить с помощью оптимизатора. 
Например, могут существовать свободные переменные, определяющие веса и смещения нейронной сети."""

# переменная, инициализированная случайными значениями
var = tf.compat.v1.get_variable('first_variable', shape=[1,3], dtype=tf.float32)
# переменная, инициализированная константами
init_val = np.array([4, 5])
# Пере- менные можно создать и как необучаемые: traiable=False
var2 = tf.compat.v1.get_variable('second_variable', shape=[1,2], dtype=tf.float32)
initializer = tf.constant(init_val)

# сщздать сеанс
sess = tf.compat.v1.Session()
# инищиализировать все пременные
sess.run(tf.compat.v1.global_variables_initializer())
print(sess.run(var))
print(sess.run(var2))

print(tf.compat.v1.global_variables())

"""Граф представляет низкоуровневые вычисления в терминах зависимостей между операциями. 
В TensorFlow сначала определяется граф, а затем создается сеанс, 
в контексте которого выполняются все включенные в граф операции."""

const1 = tf.constant(3.0, name='constant1')
var1 = tf.compat.v1.get_variable('variable1', shape=[1, 2], dtype=tf.float32)
var2 = tf.compat.v1.get_variable('variable2', shape=[1, 2], trainable=False, dtype=tf.float32)

op1 = const1 * var1
op2 = op1 + var2
op3 = tf.reduce_mean(op2)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
print(sess.run(op3))

