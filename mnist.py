import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.framework import graph_util



mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


def train():
    X = tf.placeholder(tf.float32, [None, 784])
    y_true = tf.placeholder(tf.float32, [None, 10])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    
    y = tf.nn.softmax(tf.matmul(X, W) + b, name='final')

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_true * tf.log(y), reduction_indices=[1]))

    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    # to save the model
    saver = tf.train.Saver()


    train_max_steps = 100
    eval_frequency = 10
    save_frequency = 10
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        for step in range(train_max_steps):
            batch_xs, batch_ys = mnist.train.next_batch(10)
            sess.run(train_step, feed_dict={X: batch_xs, y_true: batch_ys})

            # 存model
            if step % save_frequency == 0:
                saver.save(sess, 'model/model.ckpt')

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


def freeze_graph(input_checkpoint, output_graph):
    # 建模型
    # 'model/model.ckpt'
    X = tf.placeholder(tf.float32, [None, 784])
    y_true = tf.placeholder(tf.float32, [None, 10])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(X, W) + b, name='final')

    graph = tf.get_default_graph() # 获得默认的图
    input_graph_def = graph.as_graph_def()  # 返回一个序列化的图代表当前的图

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)
        output_graph_def = graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
                    sess=sess,
                    input_graph_def=input_graph_def,# 等于:sess.graph_def
                    output_node_names=['final'])# 如果有多个输出节点，以逗号隔开
    
    with tf.gfile.GFile(output_graph, "wb") as f: #保存模型
        f.write(output_graph_def.SerializeToString()) #序列化输出
        print("%d ops in the final graph." % len(output_graph_def.node)) #得到当前图有几个操作节点
    

if __name__ == '__main__':
    freeze_graph('model/model.ckpt', 'model_pb/test.pb')