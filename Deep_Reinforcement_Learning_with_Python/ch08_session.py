import tensorflow as tf

# sess = tf.Session()
# sess.run()

a = tf.multiply(3, 3)
print(a)

a = tf.multiply(3, 3)
with tf.Session() as sess:
    print(sess.run(a))

