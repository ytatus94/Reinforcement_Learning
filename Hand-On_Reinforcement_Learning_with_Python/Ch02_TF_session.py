import tensorflow as tf

# 定義變數
# weights = tf.Variable(tf.random_normal([3, 2], stddev=0.1), name="weights")
# 定義變數之後還要用 tf.global_variables_initializer() 來初始化

# 常數
# x = tf.constant(13)

# 佔位符 placeholder 是指定義了形態與維度的變數，不用指定數值，數值是在執行時被賦予的
# x = tf.placeholder("float", shape=None)

# 直接相依
# B 的輸入是來自 A 的輸出
# A = tf.multiply(8, 5)
# B = tf.multiply(A, 1)

# 間接相依
# B 的輸入和 A 的輸出無關
# A = tf.multiply(8, 5)
# B = tf.multiply(4, 3)

# 階段
# sess = tf.Session()

# 初始化變數
a = tf.multiply(2, 3)
print(a)

with tf.Session() as sess:
    # 執行 session
    print(sess.run(a))
