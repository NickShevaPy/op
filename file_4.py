import tensorflow as tf
"""Вычисляется сумма, разница, произведение и частное констант a и b, 
созданных методом tf.constant()"""
def main():
   x = tf.constant(4)
   y = tf.constant(5)
   a = tf.add(x, y)
   print('Addition')
   print(a.numpy())
   s = tf.subtract(x, y)
   print('Substracation')
   print(s.numpy())
   m = tf.multiply(x, y)
   print('Multiplication')
   print(m.numpy())
   d = tf.divide(x, y)
   print('Division')
   print(d.numpy())



if __name__ == "__main__":
    main()
