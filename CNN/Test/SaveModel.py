import tensorflow as tf

w1 = tf.placeholder(tf.float32, name="w1")
w2 = tf.placeholder(tf.float32, name="w2")

b1 = tf.Variable(2.0, name="bias")

sess = tf.InteractiveSession()

sess.run(tf.global_variables_initializer())

w3 = tf.add(w1, w2)
w4 = tf.multiply(w3, b1, name="op_to_restore")

feed_dict={w1:4, w2:8}

saver = tf.train.Saver()

print(sess.run(w4, feed_dict))

saver.save(sess, "./checkpoints_dir/test_model", global_step=1000)