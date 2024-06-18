import tensorflow as tf
"""Вычисляется сумма, разница констант a и b, 
созданных методом tf.constant()"""
def main():
   a = tf.constant(4)
   b = tf.constant(5)
   c = tf.add(a, b)
   print(c.numpy())
   s = tf.subtract(a, b)
   print(s.numpy())


if __name__ == "__main__":
    main()
