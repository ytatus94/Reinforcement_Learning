import tensorflow as tf

# Variables
# variable is used to store data
x = tf.Variable(13)
W = tf.Variable(tf.random_normal([500, 111], stddev=0.35), name="weights")

x = tf.Variable(1212)
# Initialize all variables in the computational graph
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(x))

# Constants
x = tf.constant(13)

# Placeholders
# placeholder is used to feed in external data
# Use feed_dict={placeholder: value} to feed data
x = tf.placeholder("float", shape=None)

# This will return an error
x = tf.placeholder("float", None)
x = x + 3
with tf.Session() as sess:
    result = sess.run(y)
    print(result)

with tf.Session() as sess:
    result = sess.run(y, feed_dict={x: 5})
    print(result)

