# Credit: Tathagat Dasgupta (https://towardsdatascience.com/deep-autoencoders-using-tensorflow-c68f075fd1a3)
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./MNIST/", one_hot=True)

latent_size = 25
input_size = 28*28
# Encoder Structure
enc1_size = 512
enc2_size = 256
enc3_size = latent_size
# Decoder Structure
dec1_size = 256
dec2_size = 512
output_size = input_size
# Learning Rate and Activation Function
lr = 0.01
act = tf.nn.relu

# Create Variables for the Weights and Biases
X = tf.placeholder(tf.float32, shape=[None, input_size])
initializer = tf.variance_scaling_initializer()

# Weights
w1 = tf.Variable(initializer([input_size, enc1_size]), dtype=tf.float32)
w2 = tf.Variable(initializer([enc1_size, enc2_size]), dtype=tf.float32)
w3 = tf.Variable(initializer([enc2_size, enc3_size]), dtype=tf.float32)
w4 = tf.Variable(initializer([enc3_size, dec1_size]), dtype=tf.float32)
w5 = tf.Variable(initializer([dec1_size, dec2_size]), dtype=tf.float32)
w6 = tf.Variable(initializer([dec2_size, output_size]), dtype=tf.float32)

# Biases
b1 = tf.Variable(tf.zeros(enc1_size))
b2 = tf.Variable(tf.zeros(enc2_size))
b3 = tf.Variable(tf.zeros(enc3_size))
b4 = tf.Variable(tf.zeros(dec1_size))
b5 = tf.Variable(tf.zeros(dec2_size))
b6 = tf.Variable(tf.zeros(output_size))

# Layers
enc1_layer = act(tf.matmul(X,w1)+b1)
enc2_layer = act(tf.matmul(enc1_layer, w2)+b2)
latent_layer = tf.matmul(enc2_layer, w3)+b3
dec1_layer = act(tf.matmul(latent_layer, w4)+b4)
dec2_layer = act(tf.matmul(dec1_layer, w5)+b5)
output_layer = tf.nn.sigmoid(tf.matmul(dec2_layer, w6)+b6)

# Loss / Optimizer / Initialization
loss = tf.reduce_mean(tf.square(output_layer-X))
optimizer = tf.train.AdamOptimizer(lr)
train = optimizer.minimize(loss)
init = tf.global_variables_initializer()

num_epoch = 25
batch_size = 32
num_test_images = 10

Train_Loss = []
Valid_Loss =[]

# Train
loss_record = 1
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(num_epoch):
        num_batches = mnist.train.num_examples // batch_size
        for iteration in range(num_batches):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(train, feed_dict={X:X_batch})

        train_loss = loss.eval(feed_dict={X:X_batch})
        print("Epoch {}/{} Training Loss: {: .4f}".format(epoch+1, num_epoch, train_loss))
        Train_Loss.append(train_loss)

        X_val, y_val = mnist.validation.next_batch(5000)
        valid_loss = loss.eval(feed_dict={X:X_val})
        print("Epoch {}/{} Validation Loss: {: .4f}".format(epoch + 1, num_epoch, valid_loss))
        Valid_Loss.append(valid_loss)
        if valid_loss < loss_record:
            saver.save(sess,"./best_valid.ckpt")

    saver.restore(sess, "./best_valid.ckpt")
    X_test, y_test = mnist.test.next_batch(10000)
    test_loss = loss.eval(feed_dict={X:X_test})

    x_axis = range(1, num_epoch+1)
    plt.plot(x_axis, Train_Loss, '-r')
    plt.plot(x_axis, Valid_Loss, '--b')
    plt.legend(['Training Loss', 'Validation Loss'])
    min_loss_idx = Valid_Loss.index(min(Valid_Loss))
    plt.plot(min_loss_idx + 1, Valid_Loss[min_loss_idx], '.k')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    #results = output_layer.eval(feed_dict={X:mnist.test.images[:num_test_images]})
    # Comparing original images with reconstructions
    #f,a = plt.subplots(2, 10, figsize=(20, 4))
    #for i in range(num_test_images):
    #    a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
    #    a[1][i].imshow(np.reshape(results[i], (28, 28)))
