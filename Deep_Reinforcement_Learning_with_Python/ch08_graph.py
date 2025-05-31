import tensorflow as tf

# Create TensorFlow graph
graph = tf.Graph()

with graph.as_default():
    z = tf.add(x, y, name="Add")

