import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


def train():
    X = tf.placeholder(tf.float32, [None, 784])
    y_true = tf.placeholder(tf.float32, [None, 10])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    
    y = tf.nn.softmax(tf.matmul(X, W) + b)

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_true * tf.log(y), reduction_indices=[1]))

    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    # to save the model
    saver = tf.train.Saver()


    eval_frequency = 100
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        for step in range(1000):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            sess.run(train_step, feed_dict={X: batch_xs, y_true: batch_ys})

            # 存model
            save_path = saver.save(sess, 'model/model.ckpt')

            if step % eval_frequency == 0:
                correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_true, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                print('Accuracy: ', sess.run(accuracy, feed_dict = {X: mnist.test.images, y_true: mnist.test.labels}))


def restore():

    X = tf.placeholder(tf.float32, [None, 784])
    y_true = tf.placeholder(tf.float32, [None, 10])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    y = tf.nn.softmax(tf.matmul(X, W) + b)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, 'model/model.ckpt')

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_true, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print('Accuracy: ', sess.run(accuracy, feed_dict = {X: mnist.test.images, y_true: mnist.test.labels}))


if __name__ == '__main__':
    restore()