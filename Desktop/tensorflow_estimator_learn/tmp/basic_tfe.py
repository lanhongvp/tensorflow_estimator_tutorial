import tensorflow as tf
import tensorflow.contrib.eager as tfe
tfe.enable_eager_execution()

two_node_tfe = tf.constant(2)
three_node_tfe = tf.constant(3)
sum_node_tfe = two_node_tfe + three_node_tfe

print('TWO_NODE_TFE',two_node_tfe)
print('THREE_NODE_TFE',three_node_tfe)
print('SUM_NODE_TFE',sum_node_tfe)




