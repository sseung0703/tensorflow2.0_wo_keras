import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

import time
import scipy.io as sio
import numpy as np
from random import shuffle

from nets import ResNet

### define path and hyper-parameter
train_dir   = '/home/dmsl/Documents/tf2.0/test'

model_name   = 'vgg16'
dataset = sio.loadmat('/home/dmsl/Documents/data/tf/cifar10_natural.mat')
dataset_len, *image_size = dataset['val_image'].shape
num_labels = np.max(dataset['train_label'])+1

batch_size = 100
trained_params = sio.loadmat('/home/dmsl/Documents/tf2.0/test/best_params.mat')


def MODEL(model_name, weight_decay, image, label, trainable, is_training):
    end_points = ResNet.model(image, trainable = trainable, is_training = is_training)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(label,end_points['Logits']))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(label, 1), tf.argmax(end_points['Logits'], 1)), tf.float32))

    tf.compat.v1.summary.scalar('loss', loss)
    return end_points, loss, accuracy

#%%    
with tf.Graph().as_default() as graph:
    ## Load Dataset
    train_image = tf.keras.backend.placeholder(shape = [None]+image_size, dtype = tf.float32)
    train_label = tf.keras.backend.placeholder(shape = [None], dtype = tf.int32)
    is_training = tf.keras.backend.placeholder(dtype = tf.uint8)

    train_label_onhot = tf.one_hot(train_label, num_labels, on_value=1.0)
 
    decay_steps = dataset_len // batch_size
    ## load Net
    train_end_points, total_loss, train_accuracy = MODEL(model_name, 0.0, train_image,
                                                         train_label_onhot, False, 0)

    val_labels   = dataset['val_label'].T
    val_images   = dataset['val_image']

    with tf.compat.v1.Session(graph = graph) as sess:
        ## ready for training , init variables create coordinator etc.
#        sess.run([v.initializer for v in set(graph.get_collection('variables')) if hasattr(v, 'initializer')])
        variables = graph.get_collection('variables')
        for v in variables:
            if trained_params.get(v.name[:-2]) is not None:
                sess.run(v.assign(trained_params[v.name[:-2]].reshape(*v.get_shape().as_list()) ))
        
        ## test in each end of epoch
        sum_val_accuracy = 0
        val_itr = dataset_len//batch_size
        for i in range(val_itr):
            val_batch = val_images[i*batch_size:(i+1)*batch_size]
            acc = sess.run(train_accuracy, feed_dict = {train_image : val_batch,
                                                        train_label : np.squeeze(val_labels[i*batch_size:(i+1)*batch_size]),
                                                        is_training : 0})
            sum_val_accuracy += acc

        print ('val_Accuracy : %.2f%%' %(sum_val_accuracy *100/val_itr))


