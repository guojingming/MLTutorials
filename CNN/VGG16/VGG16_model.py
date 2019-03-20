import tensorflow as tf

def conv2d(x, shape, padding="SAME", step=[1, 1]):
    return tf.nn.conv2d(x, shape, strides=[1, step[0], step[1], 1], padding=padding)

def maxpool(x, shape=[1, 2, 2, 1], padding="SAME", step=[2, 2]):
    return tf.nn.max_pool(x, ksize=shape, strides=[1, step[0], step[1], 1], padding=padding)

def genweights(shape):
    return tf.Variable(tf.truncated_normal(shape=shape, stddev=0.1))

def genbias(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))


convLayers = []
maxPoolLayers = []
placeHolders = []

sess = tf.InteractiveSession()

input_img = tf.placeholder(tf.float32, shape=[None, 224 * 224 * 3], name="input_img")
input_x = tf.reshape(input_img, shape=[-1, 224, 224, 3])

#1:conv1     in: 3@224*224    out: 64@224*224
conv1_w = genweights([3, 3, 3, 64])
conv1_b = genbias([64])
conv1_h = tf.nn.relu(conv2d(input_x, conv1_w) + conv1_b)

#2:conv2     in: 64@224*224   out: 64@224*224
conv2_w = genweights([3, 3, 64, 64])
conv2_b = genbias([64])
conv2_h = tf.nn.relu(conv2d(conv1_h, conv2_w) + conv2_b)

#3:maxpool1  in: 64@224*224   out: 64@112*112
maxpool1_h = maxpool(conv2_h)

#4:conv3     in: 64@112*112   out: 128@112*112
conv3_w = genweights([3, 3, 64, 128])
conv3_b = genbias([128])
conv3_h = tf.nn.relu(conv2d(maxpool1_h, conv3_w) + conv3_b)

#5:conv4     in: 128@112*112  out: 128@112*112
conv4_w = genweights([3, 3, 128, 128])
conv4_b = genbias([128])
conv4_h = tf.nn.relu(conv2d(conv3_h, conv4_w) + conv4_b)

#6:maxpool2  in: 128@112*112  out: 128@56*56
maxpool2_h = maxpool(conv4_h)

#7:conv5     in: 128@56*56    out: 256@56*56
conv5_w = genweights([3, 3, 128, 256])
conv5_b = genbias([256])
conv5_h = tf.nn.relu(conv2d(maxpool2_h, conv5_w) + conv5_b)

#8:conv6     in: 256@56*56    out: 256@56*56
conv6_w = genweights([3, 3, 256, 256])
conv6_b = genbias([256])
conv6_h = tf.nn.relu(conv2d(conv5_h, conv6_w) + conv6_b)

#9:conv7     in: 256@56*56    out: 256@56*56
conv7_w = genweights([3, 3, 256, 256])
conv7_b = genbias([256])
conv7_h = tf.nn.relu(conv2d(conv6_h, conv7_w) + conv7_b)

#10:maxppol3 in: 256@56*56    out: 256@28*28
maxpool3_h = maxpool(conv7_h)

#11:conv8    in: 256@28*28    out: 512@28*28
conv8_w = genweights([3, 3, 256, 512])
conv8_b = genbias([512])
conv8_h = tf.nn.relu(conv2d(maxpool3_h, conv8_w) + conv8_b)

#12:conv9    in: 512@28*28    out: 512@28*28
conv9_w = genweights([3, 3, 512, 512])
conv9_b = genbias([512])
conv9_h = tf.nn.relu(conv2d(conv8_h, conv9_w) + conv9_b)

#13:conv10   in: 512@28*28    out: 512@28*28
conv10_w = genweights([3, 3, 512, 512])
conv10_b = genbias([512])
conv10_h = tf.nn.relu(conv2d(conv9_h, conv10_w) + conv10_b)

#14:maxpool4 in: 512@28*28    out: 512@14*14
maxpool4_h = maxpool(conv10_h)

#15:conv11   in: 512@14*14    out: 512@14*14
conv11_w = genweights([3, 3, 512, 512])
conv11_b = genbias([512])
conv11_h = tf.nn.relu(conv2d(maxpool4_h, conv11_w) + conv11_b)

#16:conv12   in: 512@14*14    out: 512@14*14
conv12_w = genweights([3, 3, 512, 512])
conv12_b = genbias([512])
conv12_h = tf.nn.relu(conv2d(conv11_h, conv12_w) + conv12_b)

#17:conv13   in: 512@14*14    out: 512@14*14
conv13_w = genweights([3, 3, 512, 512])
conv13_b = genbias([512])
conv13_h = tf.nn.relu(conv2d(conv12_h, conv13_w) + conv13_b)

#18:maxpool5 in: 512@14*14    out: 512@7*7
maxpool5_h = maxpool(conv13_h)

#19:fc1      in: 512@7*7      out: 4096@1*1
fc1_w = genweights([7 * 7 * 512, 4096])
fc1_b = genbias([4096])

maxpool5_h_flat = tf.reshape(maxpool5_h, [-1, 7 * 7 * 512])
fc1_h = tf.nn.relu(tf.matmul(maxpool5_h_flat, fc1_w) + fc1_b)
# dropout
keep_prob = tf.placeholder(tf.float32)
fc1_h_drop = tf.nn.dropout(fc1_h, keep_prob)

#20:fc2      in: 4096@1*1     out: 4096@1*1
fc2_w = genweights([4096, 4096])
fc2_b = genbias([4096])

fc2_h = tf.nn.relu(tf.matmul(fc1_h_drop, fc2_w) + fc2_b)
# dropout
fc2_h_drop = tf.nn.dropout(fc2_h, keep_prob)

#21:fc3      in: 4096@1*1     out: 1000@1*1
fc3_w = genweights([4096, 91])
fc3_b = genbias([91])
fc3_h = tf.matmul(fc2_h_drop, fc3_w) + fc3_b

#22:res      in: 1000@1*1     out: softmax res
y_conv = tf.nn.softmax(fc3_h)
y_ = tf.placeholder(tf.float32, [None, 91])

# model training
#cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
#train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#correct_prediction = tf.equal(tf.arg_max(y_conv, 1), tf.arg_max(y_, 1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#sess.run(tf.global_variables_initializer())


#for i in range(20000):
#    batch = mnist.train.next_batch(50)
#    if i % 100 == 0:
#        train_accuacy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
#        print("step %d, training accuracy %g"%(i, train_accuacy))
#    train_step.run(feed_dict = {x: batch[0], y_: batch[1], keep_prob: 0.5})
