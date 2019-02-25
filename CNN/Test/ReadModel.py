import tensorflow as tf

sess = tf.InteractiveSession()

saver = tf.train.import_meta_graph("./checkpoints_dir/test_model-1000.meta")
saver.restore(sess, tf.train.latest_checkpoint("./checkpoints_dir"))

graph = tf.get_default_graph()
w1 = graph.get_tensor_by_name("w1:0")
w2 = graph.get_tensor_by_name("w2:0")

feed_dict = {w1:13.0, w2:17.0}

op_to_restore = graph.get_tensor_by_name("op_to_restore:0")

add_on_op = tf.multiply(op_to_restore, 2)

print(sess.run(add_on_op, feed_dict))