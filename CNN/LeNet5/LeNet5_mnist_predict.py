import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data

import numpy as np
import scipy.misc
import cv2

epoch_count = 20000

saver = tf.train.import_meta_graph("./checkpoint_dir/mnist_model{0}.meta".format(epoch_count))

sess = tf.InteractiveSession()

saver.restore(sess, tf.train.latest_checkpoint("./checkpoint_dir"))

graph = tf.get_default_graph()

variable_names = [v.name for v in graph.get_operations()]

predict_op = graph.get_tensor_by_name("predict:0")
input_x = graph.get_tensor_by_name("input_x:0")

# mnist = input_data.read_data_sets("./LeNet5/MNIST_data/", one_hot=True)

cv2.namedWindow("Picture")

pic_count = 500


for i in range(0, pic_count):

    #batch = mnist.train.next_batch(1)
    #img = batch[0].reshape((28, 28))

    img = cv2.imread("./mnist_train_pic/out{0}.bmp".format(i), cv2.IMREAD_GRAYSCALE)
    array = np.array(img)
    #scipy.misc.imsave("./mnist_train_pic/out{0}.bmp".format(i), img)
    feed_dict = {input_x: array.reshape(1, 784)}
    res_array = sess.run(predict_op, feed_dict)[0].tolist()
    predict_result = res_array.index(max(res_array));
    cv2.imshow("Picture", img)
    print("Result: {0}".format(predict_result))
    cv2.waitKey(0)
