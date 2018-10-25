import tensorflow as tf

input_placeholder = tf.placeholder(tf.int32)
three_node = tf.constant(3)
sum_node = input_placeholder + three_node
sess = tf.Session()

print('THREE_NODE',sess.run(three_node))
print('SUM_NODE',sess.run(sum_node))

