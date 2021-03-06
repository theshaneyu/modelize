import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.framework import graph_util
from random import sample
from time import time
import os
import subprocess



mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


def train():
    X = tf.placeholder(tf.float32, [None, 784], name='input_node')
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
                print('Accuracy: ', sess.run(accuracy, feed_dict={X: mnist.test.images, y_true: mnist.test.labels}))

def save_pb_with_freeze_graph(input_checkpoint, output_graph):
    """[讀取ckpt檔，輸出pb檔]
    函数save_pb_with_freeze_graph中，最重要的就是要确定“指定输出的节点名称”，这个节点名称必须是原模型中存在的节点，对于freeze操作，我们需要定义输出结点的名字。
    因为网络其实是比较复杂的，定义了输出结点的名字，那么freeze的时候就只把输出该结点所需要的子图都固化下来，其他无关的就舍弃掉。
    因为我们freeze模型的目的是接下来做预测。所以，output_node_names一般是网络模型最后一层输出的节点名称，或者说就是我们预测的目标。
    
    Arguments:
        input_checkpoint {[str]} -- [輸入ckpt model的檔案夾]
        output_graph {[str]} -- [輸出的pb檔]
    """

    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)

    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)
        
        # 用convert_variables_to_constants將模型模型持久化，這個函數需指定需要固定的節點名稱，
        # 以mnist為例，需要固定的節點只有一個，就是最後一層softmax，
        # convert_variables_to_constants會自動把得到這個節點數值所需要的節點都固定。
        output_graph_def = graph_util.convert_variables_to_constants(sess=sess,
                                                                     input_graph_def=sess.graph_def,
                                                                     output_node_names=['final']) # 如果有多个输出节点，以逗号隔开
    
        with tf.gfile.GFile(output_graph, "wb") as f: #保存模型
            f.write(output_graph_def.SerializeToString()) #序列化输出
        print("%d ops in the final graph." % len(output_graph_def.node)) #得到当前图有几个操作节点

def save_pb_with_builder(input_checkpoint, output_path):

    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
    
    with tf.Session() as sess:
        
        saver.restore(sess, input_checkpoint)
        
        if os.path.exists(output_path):
            subprocess.run(['rm', '-rf', output_path])
        builder = tf.saved_model.builder.SavedModelBuilder(output_path)
        
        
        input_image_tensor = sess.graph.get_tensor_by_name("input_node:0")
        output_tensor_name = sess.graph.get_tensor_by_name("final:0")
        

        model_signature = tf.saved_model.signature_def_utils.build_signature_def(
            inputs={
                'input_node': tf.saved_model.utils.build_tensor_info(input_image_tensor)
            },
            outputs={
                'output_node': tf.saved_model.utils.build_tensor_info(output_tensor_name)
            },
            method_name='mnist_builder_pb')
        

        builder.add_meta_graph_and_variables(sess=sess,
                                             tags=['mnist_builder_pb'],
                                             signature_def_map={'mnist_builder_pb_signature': model_signature})
        builder.save()
    
def load_ckpt_file(input_checkpoint, to_print=False):
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
    
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)
        
        input_image_tensor = sess.graph.get_tensor_by_name("input_node:0")
        output_tensor_name = sess.graph.get_tensor_by_name("final:0")
        
        # output = sess.run(output_tensor_name, feed_dict={input_image_tensor: mnist.test.images})
        output = evaluate_sess_run_time(sess, input_image_tensor, output_tensor_name)
    if to_print:
        print_ten_prediction(output)

def load_pb_file(pb_file_path, to_print=False):
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        with open(pb_file_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def, name="")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # 定義輸入的tensor名稱（input tensor是placeholder）
            input_image_tensor = sess.graph.get_tensor_by_name("input_node:0")

            # 定義输出的tensor名稱
            output_tensor_name = sess.graph.get_tensor_by_name("final:0")
            
            
            # output = sess.run(output_tensor_name, feed_dict={input_image_tensor: mnist.test.images})
            output = evaluate_sess_run_time(sess, input_image_tensor, output_tensor_name)
    if to_print:
        print_ten_prediction(output)

def load_pb_produced_by_builder(builder_pb_dir, to_print=False):
    with tf.Session() as sess:
        tf.saved_model.loader.load(sess, ['mnist_builder_pb'], builder_pb_dir)
        sess.run(tf.tables_initializer())
        # sess.run(tf.global_variables_initializer())

        # for item in tf.get_default_graph().as_graph_def().node:
        #     print(item.name)

        input_image_tensor = sess.graph.get_tensor_by_name('input_node:0')
        output_tensor_name = sess.graph.get_tensor_by_name("final:0")

        # output = sess.run(output_tensor_name, feed_dict={input_image_tensor: mnist.test.images})
        output = evaluate_sess_run_time(sess, input_image_tensor, output_tensor_name)
    if to_print:
        print_ten_prediction(output)

def evaluate_sess_run_time(sess, in_node, out_node):
    # 跑完 mnist.test.images 要花的時間
    start = time()

    sess.run(out_node, feed_dict={in_node: mnist.test.images})

    end = time()
    print('花', end - start)
    
def evaluate_run_time(ckpt_file, pb_file, builder_pb_dir):
    # start = time()
    # load_ckpt_file(ckpt_file)
    # end = time()
    # print('ckpt檔花', end - start)

    # start = time()
    # load_pb_file(pb_file)
    # end = time()
    # print('一般pb檔花', end - start)

    start = time()
    load_pb_produced_by_builder(builder_pb_dir)
    end = time()
    print('使用builder存的pb檔花', end - start)

def print_ten_prediction(output):
    # for n in sample(range(10000), 10):
    for n in range(5):
        print('[正確]')
        print(mnist.test.labels[n].tolist().index(1))
        print('[預測]')
        print(output[n].tolist().index(float(max(output[n]))))
        print('-----------------------')




if __name__ == '__main__':
    # train()

    input_ckpt_file = 'big_ckpt/inception_resnet_v2_2016_08_30.ckpt'
    output_pb_file = 'big_pb/inception_resnet_v2_2016_08_30.pb'
    save_pb_with_freeze_graph('model/model.ckpt', 'model_pb/freeze_graph.pb')
    # save_pb_with_builder('model/model.ckpt', 'model_pb_builder')
    
    # load_ckpt_file('model/model.ckpt', to_print=True)
    # load_pb_file('model_pb/test1.pb', to_print=True)
    # load_pb_produced_by_builder('model_pb_builder', to_print=True)


    # load_ckpt_file('model/model.ckpt')
    # load_pb_file('model_pb/test1.pb')
    # load_pb_produced_by_builder('model_pb_builder')

    
    # evaluate_run_time('model/model.ckpt', './model_pb/test1.pb', 'model_pb_builder')

    
