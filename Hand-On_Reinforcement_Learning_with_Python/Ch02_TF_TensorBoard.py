import tensorflow as tf

a = tf.constant(5)
b = tf.constant(4)
c = multiply(a, b)
d = tf.constant(2)
e = tf.constant(3)
f = tf.multiply(d, e)
g = tf.add(c, f)

with tf.Session() as sess:
    # 把運算圖寫入到 output 中
    writer = tf.summary.FileWriter("output", sess.graph)
    print(sess.run(g))
    writer.close()
