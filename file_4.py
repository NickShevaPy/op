import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import numpy as np
"""Вычисляется сумма, разница, произведение и частное констант a и b, 
созданных методом tf.constant()"""

# Create a constant tensor
x = tf.constant(8)
y = tf.constant(5)
# make session
sess = tf.compat.v1.Session()
# find add
a = tf.add(x, y)
print(f'Addition: {sess.run(a)}')
# find sub
s = tf.subtract(x, y)
print(f'Substracation: {sess.run(s)}')
# find mul
m = tf.multiply(x, y)
print(f'Multiplication: {sess.run(m)}')
# find div
d = tf.divide(x, y)
print(f'Division: {sess.run(d)}')



# Create a constant tensor
tensor = tf.constant([1, 2, 3])
# Create a session
session = tf.compat.v1.Session()
# Run the session to find the first element
result = session.run(tensor[0])
# Print the result
print(result)



