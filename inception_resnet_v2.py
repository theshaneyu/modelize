# https://blog.csdn.net/Wayne2019/article/details/78109357

import tensorflow as tf
slim = tf.contrib.slim
from PIL import Image
from nets.inception_resnet_v2 import *
import numpy as np
from preprocessing import inception_preprocessing
import matplotlib.pyplot as plt
from datasets import imagenet
from time import time
from tensorflow.python.framework import graph_util


image_size = inception_resnet_v2.default_image_size # 299


def load_ckpt(to_save_pb=False):
    tf.reset_default_graph()

    checkpoint_file = 'big_ckpt/inception_resnet_v2_2016_08_30.ckpt'
    image = tf.image.decode_jpeg(tf.read_file('pics/dog.jpg'), channels=3) # ['dog.jpg', 'panda.jpg']
    processed_image = inception_preprocessing.preprocess_image(image, image_size, image_size, is_training=False,) # 这个函数做了裁剪，缩放和归一化等
    processed_images  = tf.expand_dims(processed_image, 0)


    arg_scope = inception_resnet_v2_arg_scope() # Creates the Inception Resnet V2 model.
    with slim.arg_scope(arg_scope):
        logits, end_points = inception_resnet_v2(processed_images, is_training=False)

    probabilities = tf.nn.softmax(logits, name='final')

    saver = tf.train.Saver()

    if to_save_pb:
        convert_ckpt_to_pb(saver)
        return exit(0)

    with tf.Session() as sess:
        saver.restore(sess, checkpoint_file)

        start = time()
        # predict_values, logit_values = sess.run([end_points['Predictions'], logits])
        image2, network_inputs, probabilities2 = sess.run([image,
                                                        processed_images,
                                                        probabilities])
        end = time()

    print('ckpt time cost', end - start) # 2.4201557636260986


# 儲存 pb 用
def convert_ckpt_to_pb(saver):
    input_ckpt = 'big_ckpt/inception_resnet_v2_2016_08_30.ckpt'
    output_pb = 'big_pb/inception_resnet_v2_2016_08_30.pb'

    with tf.Session() as sess:
        saver.restore(sess, input_ckpt)

        output_graph_def = graph_util.convert_variables_to_constants(sess=sess,
                                                                     input_graph_def=sess.graph_def,
                                                                     output_node_names=['final']) # 如果有多个输出节点，以逗号隔开
    
        with tf.gfile.GFile(output_pb, "wb") as f: #保存模型
            f.write(output_graph_def.SerializeToString()) #序列化输出
        print("%d ops in the final graph." % len(output_graph_def.node)) #得到当前图有几个操作节点


def load_pb(pb_file_path):
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        with open(pb_file_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def, name="")
        
        # with tf.Session() as sess:
        #     sess.run(tf.global_variables_initializer())

        #     # 定義輸入的tensor名稱（input tensor是placeholder）
        #     input_image_tensor = sess.graph.get_tensor_by_name("input_node:0")
        #     # 定義输出的tensor名稱
        #     output_tensor_name = sess.graph.get_tensor_by_name("final:0")
            
        #     output = sess.run(output_tensor_name, feed_dict={input_image_tensor: mnist.test.images})

        start = time()
        with tf.Session() as sess:
            # saver.restore(sess, checkpoint_file)
            sess.run(tf.global_variables_initializer())

            # predict_values, logit_values = sess.run([end_points['Predictions'], logits])
            image2, network_inputs, probabilities2 = sess.run([image,
                                                            processed_images,
                                                            probabilities])
        end = time()


if __name__ == '__main__':
    load_ckpt(to_save_pb=False)
    # load_pb('big_pb/inception_resnet_v2_2016_08_30.pb')
        




# ---------------------------------------------- 不必要的

# print(network_inputs.shape)
# print(probabilities2.shape)
# probabilities2 = probabilities2[0,:]
# sorted_inds = [i[0] for i in sorted(enumerate(-probabilities2),
#                                     key=lambda x:x[1])]    


# # 顯示原圖
# plt.figure()
# plt.imshow(image2)
# plt.suptitle("Original image", fontsize=14, fontweight='bold')
# plt.axis('off')
# plt.show()

# # 顯示輸入模型，preprocessing 過的圖
# plt.imshow(network_inputs[0,:,:,:])
# plt.suptitle("Resized, Cropped and Mean-Centered inputs to network",
#              fontsize=14, fontweight='bold')
# plt.axis('off')
# plt.show()


# # 打印 softmax output
# names = imagenet.create_readable_names_for_imagenet_labels()
# for i in range(5):
#     index = sorted_inds[i]
#     print(index)
#     # 打印top5的预测类别和相应的概率值。
#     print('Probability %0.2f => [%s]' % (probabilities2[index], names[index+1]))








# '''https://github.com/tensorflow/models/blob/master/research/slim/train_image_classifier.py'''
# def _get_variables_to_train():
#     """Returns a list of variables to train.
#     Returns:
#       A list of variables to train by the optimizer.
#     """
#     trainable_scopes = 'InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits'

#     if trainable_scopes is None:
#       return tf.trainable_variables()
#     else:
#       scopes = [scope.strip() for scope in trainable_scopes.split(',')]

#     variables_to_train = []
#     for scope in scopes:
#       variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
#       variables_to_train.extend(variables)
#     return variables_to_train

# '''
# 一些关于inception_resnet_v2变量的测试，在理解模型代码和迁移学习中很有用
# '''
# exx = tf.trainable_variables()
# print(type(exx))
# print(exx[0])
# print(exx[-1])
# print(exx[-2])
# print(exx[-3])
# print(exx[-4])
# print(exx[-5])
# print(exx[-6])
# print(exx[-7])
# print(exx[-8])
# print(exx[-9])
# print(exx[-10])

# print('###############################################################')
# variables_to_train = _get_variables_to_train()
# print(variables_to_train)

# print('###############################################################')
# exclude = ['InceptionResnetV2/Logits', 'InceptionResnetV2/AuxLogits']
# variables_to_restore = slim.get_variables_to_restore(exclude = exclude)
# print(variables_to_restore[0])
# print(variables_to_restore[-1])

# print('###############################################################')
# exclude = ['InceptionResnetV2/Logits']
# variables_to_restore = slim.get_variables_to_restore(exclude = exclude)
# print(variables_to_restore[0])
# print(variables_to_restore[-1])
