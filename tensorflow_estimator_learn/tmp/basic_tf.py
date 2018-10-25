import tensorflow as tf

two_node = tf.constant(2)
three_node = tf.constant(3)
sum_node = two_node + three_node

sess = tf.Session()
two_node,three_node,sum_node = \
sess.run([two_node,three_node,sum_node])

print('TWO_NODE',two_node)
print('THREE_NODE',three_node)
print('SUM_NODE',sum_node)

# print('TWO_NODE',two_node)
# print('THREEE_NODE',three_node)
# print('SUM_NODE',sum_node)




