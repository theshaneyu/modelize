import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.framework import graph_util



# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


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
    """函数freeze_graph中，最重要的就是要确定“指定输出的节点名称”，这个节点名称必须是原模型中存在的节点，对于freeze操作，我们需要定义输出结点的名字。
    因为网络其实是比较复杂的，定义了输出结点的名字，那么freeze的时候就只把输出该结点所需要的子图都固化下来，其他无关的就舍弃掉。
    因为我们freeze模型的目的是接下来做预测。所以，output_node_names一般是网络模型最后一层输出的节点名称，或者说就是我们预测的目标。
    
    Arguments:
        input_checkpoint {[str]} -- [輸入ckpt model的檔案夾]
        output_graph {[str]} -- [輸出的pb檔]
    """

    # # 建模型
    # X = tf.placeholder(tf.float32, [None, 784])
    # y_true = tf.placeholder(tf.float32, [None, 10])
    # W = tf.Variable(tf.zeros([784, 10]))
    # b = tf.Variable(tf.zeros([10]))
    # y = tf.nn.softmax(tf.matmul(X, W) + b, name='final')

    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)

    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)
        # 用convert_variables_to_constants將模型模型持久化，這個函數需指定需要固定的節點名稱，
        # 以mnist為例，需要固定的節點只有一個，就是最後一層softmax，
        # convert_variables_to_constants會自動把得到這個節點數值所需要的節點都固定。

        output_graph_def = graph_util.convert_variables_to_constants(
                    sess=sess,
                    input_graph_def=sess.graph_def,
                    output_node_names=['final']) # 如果有多个输出节点，以逗号隔开
    
        with tf.gfile.GFile(output_graph, "wb") as f: #保存模型
            f.write(output_graph_def.SerializeToString()) #序列化输出
        print("%d ops in the final graph." % len(output_graph_def.node)) #得到当前图有几个操作节点
    

def load_pb_file(pb_file_path):
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        with open(pb_file_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def, name="")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
        
        # for item in sess.graph.get_operations():
        #     print(item.values())



            # 定义输入的张量名称,对应网络结构的输入张量
            # input:0作为输入图像,keep_prob:0作为dropout的参数,测试时值为1,is_training:0训练参数
            input_image_tensor = sess.graph.get_tensor_by_name("input:0")
            input_keep_prob_tensor = sess.graph.get_tensor_by_name("keep_prob:0")
            input_is_training_tensor = sess.graph.get_tensor_by_name("is_training:0")

            # 定义输出的张量名称
            output_tensor_name = sess.graph.get_tensor_by_name("InceptionV3/Logits/SpatialSqueeze:0")

            # 读取测试图片
            im=read_image(image_path,resize_height,resize_width,normalization=True)
            im=im[np.newaxis,:]
            # 测试读出来的模型是否正确，注意这里传入的是输出和输入节点的tensor的名字，不是操作节点的名字
            # out=sess.run("InceptionV3/Logits/SpatialSqueeze:0", feed_dict={'input:0': im,'keep_prob:0':1.0,'is_training:0':False})
            out=sess.run(output_tensor_name, feed_dict={input_image_tensor: im,
                                                        input_keep_prob_tensor:1.0,
                                                        input_is_training_tensor:False})
            print("out:{}".format(out))
            score = tf.nn.softmax(out, name='pre')
            class_id = tf.argmax(score, 1)
            print "pre class_id:{}".format(sess.run(class_id))
        

if __name__ == '__main__':
    # freeze_graph('model/model.ckpt', 'model_pb/test1.pb')
    load_pb_file('./model_pb/test1.pb')