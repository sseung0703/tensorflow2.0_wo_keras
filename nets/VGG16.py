import tensorflow as tf
import nets.tf_layers as tf_layers

def model(image, trainable = False, is_training = 0):
    end_points = {}
    weight_initializer = tf.keras.initializers.VarianceScaling()
    bias_initializer = tf.keras.initializers.Zeros()
    is_training = tf.equal(is_training, 1)
    with tf.name_scope('vgg16'):
        conv = image
        with tf.name_scope('block0'):
            for i in range(1):
                conv = tf_layers.Conv2d(conv, 32, [3,3], 1, weight_initializer, bias_initializer,
                                        weight_regulerization  = 'l2',
                                        padding = 'SAME', activation_fn = tf.nn.relu,
                                        scope='conv%d'%i, trainable=trainable)
            conv = tf.nn.max_pool(conv, [1,2,2,1], [1,2,2,1], padding = 'SAME', name='pool')
        with tf.name_scope('block1'):
            for i in range(1):
                conv = tf_layers.Conv2d(conv, 64, [3,3], 1, weight_initializer, bias_initializer,
                                        weight_regulerization  = 'l2',
                                        padding = 'SAME', activation_fn = tf.nn.relu,
                                        scope='conv%d'%i, trainable=trainable)
            conv = tf.nn.max_pool(conv, [1,2,2,1], [1,2,2,1], padding = 'SAME', name='pool') 
        with tf.name_scope('block2'):
            for i in range(1):
                conv = tf_layers.Conv2d(conv, 128, [3,3], 1, weight_initializer, bias_initializer,
                                        weight_regulerization  = 'l2',
                                        padding = 'SAME', activation_fn = tf.nn.relu,
                                        scope='conv%d'%i, trainable=trainable)
            conv = tf.nn.max_pool(conv, [1,2,2,1], [1,2,2,1], padding = 'SAME', name='pool')
        with tf.name_scope('block3'):
            for i in range(1):
                conv = tf_layers.Conv2d(conv, 256, [3,3], 1, weight_initializer, bias_initializer,
                                        weight_regulerization  = 'l2',
                                        padding = 'SAME', activation_fn = tf.nn.relu,
                                        scope='conv%d'%i, trainable=trainable)
        with tf.name_scope('block4'):
            for i in range(1):
                conv = tf_layers.Conv2d(conv, 512, [3,3], 1,  
                            weight_initializer, bias_initializer,
                            weight_regulerization  = 'l2',
                            padding = 'SAME', activation_fn = tf.nn.relu,
                            scope='conv%d'%i, trainable=trainable)
        
        fc = tf_layers.Flatten(conv)
        fc = tf_layers.Dense(fc, 1024, 
                             weight_initializer, bias_initializer,
                             weight_regulerization  = 'l2',
                             activation_fn = tf.nn.relu,
                             scope='fc0', trainable=trainable)
        
        fc = tf.cond(is_training, lambda : tf.nn.dropout(fc, 0),
                                  lambda : fc)
#        fc = tf_layers.Dense(fc, 1024, 
#                             weight_initializer, bias_initializer,
#                             weight_regulerization  = 'l2',
#                             activation_fn = tf.nn.relu,
#                             scope='fc0', trainable=trainable)
#        
#        fc = tf.cond(is_training, lambda : tf.nn.dropout(fc, 0),
#                                  lambda : fc)
        logits = tf_layers.Dense(fc, 100, 
                                 weight_initializer, bias_initializer,
                                 weight_regulerization  = 'l2',
                                 activation_fn = None,
                                 scope='fc2', trainable=trainable)
        
    end_points['Logits'] = logits
    return end_points