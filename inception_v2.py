import tensorflow as tf
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import scipy.io as scio
 
 
model_dir='F:/fqh/models-master/tutorials/image/imagenet/2015'
image = 'F:/fqh/models-master/tutorials/image/imagenet/data_set/face/faces95_72_20_180-200jpgfar-close/'
 
target_path=image+'wjhugh/'

class NodeLookup(object):
    def __init__(self, label_lookup_path=None, uid_lookup_path=None):
        if not label_lookup_path:
            label_lookup_path = os.path.join(
                    model_dir, 'imagenet_2012_challenge_label_map_proto.pbtxt')
        if not uid_lookup_path:
            uid_lookup_path = os.path.join(
                    model_dir, 'imagenet_synset_to_human_label_map.txt')
        self.node_lookup = self.load(label_lookup_path, uid_lookup_path)
 
    def load(self, label_lookup_path, uid_lookup_path):
        if not tf.gfile.Exists(uid_lookup_path):
            tf.logging.fatal('File does not exist %s', uid_lookup_path)
        if not tf.gfile.Exists(label_lookup_path):
            tf.logging.fatal('File does not exist %s', label_lookup_path)
 
        proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
        
        uid_to_human = {}
        for line in proto_as_ascii_lines:
 
            line = line.strip('\n')
 
            parse_items = line.split('\t')
 
            uid = parse_items[0]
 
            human_string = parse_items[1]
 
            uid_to_human[uid] = human_string
            
 
 
        proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
 
        node_id_to_uid = {}
        for line in proto_as_ascii:
 
            if line.startswith('  target_class:'):
 
                target_class = int(line.split(': ')[1])
            if line.startswith('  target_class_string:'):
 
                target_class_string = line.split(': ')[1]
 
                node_id_to_uid[target_class] = target_class_string[1:-2]
    
 
        node_id_to_name = {}
        for key, val in node_id_to_uid.items():
 
            if val not in uid_to_human:
                tf.logging.fatal('Failed to locate: %s', val)
 
            name = uid_to_human[val]
 
            node_id_to_name[key] = name
    
        return node_id_to_name
 
 
    def id_to_string(self, node_id):
 
        if node_id not in self.node_lookup:
            return ''
        return self.node_lookup[node_id]
 
 
def create_graph():
    with tf.gfile.FastGFile(os.path.join(
            model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
 
create_graph()
 
 
with tf.Session() as sess:
 
    softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
    
 
    for root, dirs, files in os.walk(target_path):
        for file in files:
            # print(file)
            img_path = target_path+file
            image_data = tf.gfile.FastGFile(img_path, 'rb').read()
            fc_tensor = sess.graph.get_tensor_by_name('pool_3:0')
            pool_1 = sess.run(fc_tensor,{'DecodeJpeg/contents:0': image_data})
    
            # print(pool_1)
            img_path=img_path[:len(img_path)-4]
            #print(img_path)
            scio.savemat(img_path+'.mat', {"pool_1": pool_1})
