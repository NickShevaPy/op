import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import numpy as np

"""Местозаполнитель (placeholder) – это тензор, который получает значение во время выполнения."""

a = tf.compat.v1.placeholder(shape=(1, 3), dtype=tf.float32)
b = tf.constant([1, 2, 3], dtype=tf.float32)
c = a + b
sess = tf.compat.v1.Session()
print(sess.run(c, feed_dict={a:[[0.1, 0.1, 0.1]]}))

# NB: размер по первому измерению равен 'None', т. е. длина может быть любой
a = tf.compat.v1.placeholder(shape=(None, 3), dtype=tf.float32)
b = tf.compat.v1.placeholder(shape=(None, 3), dtype=tf.float32)
c = a + b
session = tf.compat.v1.Session()
res = session.run(c, feed_dict={a: [[0.1, 0.1, 0.1]], b: [[2, 3, 4]]})
print(res)

# Define a placeholder and operation
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 1], name='x')

y = tf.add(x, 2)  # Operation with the placeholder

sess = tf.compat.v1.Session()
# Correctly feed the placeholder with data
res =sess.run(y, feed_dict={x:[[1], [2], [3]]})

print("Result:", res)  # Expected output: [[3], [4], [5]]

