"""
Created on Fri Sep 29 16:25:16 2017

@author: wayne
"""

'''
我们用的是tf1.2，最新的tf1.3地址是
https://github.com/tensorflow/models/tree/master/research/slim

http://geek.csdn.net/news/detail/126133
如何用TensorFlow和TF-Slim实现图像分类与分割

https://www.2cto.com/kf/201706/649266.html
【Tensorflow】辅助工具篇——tensorflow slim(TF-Slim)介绍

https://stackoverflow.com/questions/39582703/using-pre-trained-inception-resnet-v2-with-tensorflow
The Inception networks expect the input image to have color channels scaled from [-1, 1]. As seen here.
You could either use the existing preprocessing, or in your example just scale the images yourself: im = 2*(im/255.0)-1.0 before feeding them to the network.
Without scaling the input [0-255] is much larger than the network expects and the biases all work to very strongly predict category 918 (comic books).
'''

import tensorflow as tf
slim = tf.contrib.slim
from PIL import Image
from nets.inception_resnet_v2 import *
import numpy as np
from preprocessing import inception_preprocessing
import matplotlib.pyplot as plt
from datasets import imagenet  #注意需要用最新版tf中的对应文件，否则http地址是不对的

tf.reset_default_graph()

checkpoint_file = 'big_ckpt/inception_resnet_v2_2016_08_30.ckpt'
image = tf.image.decode_jpeg(tf.read_file('pics/dog.jpg'), channels=3) #['dog.jpg', 'panda.jpg']

image_size = inception_resnet_v2.default_image_size #  299

'''这个函数做了裁剪，缩放和归一化等'''
processed_image = inception_preprocessing.preprocess_image(image, 
                                                        image_size, 
                                                        image_size,
                                                        is_training=False,)
processed_images  = tf.expand_dims(processed_image, 0)

'''Creates the Inception Resnet V2 model.'''
arg_scope = inception_resnet_v2_arg_scope()
with slim.arg_scope(arg_scope):
  logits, end_points = inception_resnet_v2(processed_images, is_training=False)   

probabilities = tf.nn.softmax(logits)

saver = tf.train.Saver()


with tf.Session() as sess:
    saver.restore(sess, checkpoint_file)

    #predict_values, logit_values = sess.run([end_points['Predictions'], logits])
    image2, network_inputs, probabilities2 = sess.run([image,
                                                       processed_images,
                                                       probabilities])

    print(network_inputs.shape)
    print(probabilities2.shape)
    probabilities2 = probabilities2[0,:]
    sorted_inds = [i[0] for i in sorted(enumerate(-probabilities2),
                                        key=lambda x:x[1])]    


# 显示下载的图片
plt.figure()
plt.imshow(image2)#.astype(np.uint8))
plt.suptitle("Original image", fontsize=14, fontweight='bold')
plt.axis('off')
plt.show()

# 显示最终传入网络模型的图片
plt.imshow(network_inputs[0,:,:,:])
plt.suptitle("Resized, Cropped and Mean-Centered inputs to network",
             fontsize=14, fontweight='bold')
plt.axis('off')
plt.show()

names = imagenet.create_readable_names_for_imagenet_labels()
for i in range(5):
    index = sorted_inds[i]
    print(index)
    # 打印top5的预测类别和相应的概率值。
    print('Probability %0.2f => [%s]' % (probabilities2[index], names[index+1]))





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
