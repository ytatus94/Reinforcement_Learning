import tensorflow as tf

# direct dependency
a = tf.multiply(8, 5)
b = tf.multiply(a, 1) # b depends on the output of a

# indirect dependency
a = tf.multiply(8, 5)
b = tf.multiply(4, 3) # b doesn't depend on a

