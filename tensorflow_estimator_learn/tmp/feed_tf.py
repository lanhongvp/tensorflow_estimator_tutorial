import tensorflow as tf 

input_placeholder = tf.placeholder(tf.int32) 
sess = tf.Session() 
input = sess.run(\
input_placeholder, feed_dict={input_placeholder: 2})

print('INPUT',input)

