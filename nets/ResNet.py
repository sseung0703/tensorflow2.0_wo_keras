import tensorflow as tf
import nets.tf_layers as tf_layers

def model(image, trainable = False, is_training = 0):
    end_points = {}
    weight_initializer = tf.keras.initializers.VarianceScaling()
    bias_initializer = tf.keras.initializers.Zeros()
    is_training = tf.equal(is_training, 1)
    with tf.name_scope('ResNet18'):
        conv = tf_layers.Conv2d(image, 16, [3,3], 2, weight_initializer, bias_initializer,
                                weight_regulerization  = 'l2',
                                padding = 'SAME', activation_fn = None,
                                scope='conv0', trainable=trainable)
        conv = tf_layers.BatchNorm(conv, is_training, trainable = trainable,
                                   center=True, scale =True, activation_fn= tf.nn.relu,
                                   scope='bn0')
        with tf.name_scope('block0'):
            for i in range(1):
                conv = residual(conv, 64, 2, trainable = False, is_training= is_training, scope = 'Resblock0.%d'%i)

        with tf.name_scope('block1'):
            for i in range(1):
                conv = residual(conv,128, 2, trainable = False, is_training= is_training, scope = 'Resblock1.%d'%i)

        with tf.name_scope('block2'):
            for i in range(1):
                conv = residual(conv,256, 1, trainable = False, is_training= is_training, scope = 'Resblock2.%d'%i)
        
        fc = tf.reduce_mean(conv, [1,2])
        logits = tf_layers.Dense(fc, 100, 
                                 weight_initializer, bias_initializer,
                                 weight_regulerization  = 'l2',
                                 activation_fn = None,
                                 scope='fc2', trainable=trainable)
        
    end_points['Logits'] = logits
    return end_points

def residual(x, depth, stride, activation_fn = tf.nn.relu, trainable = False, is_training= False, reuse=False, scope=None):
    weight_initializer = tf.keras.initializers.VarianceScaling()
    bias_initializer = tf.keras.initializers.Zeros()
    with tf.name_scope(scope):
        conv = tf_layers.Conv2d(x, depth, [3,3], stride, weight_initializer, bias_initializer,
                                        weight_regulerization  = 'l2',
                                        padding = 'SAME', activation_fn = None,
                                        scope='conv0', trainable=trainable)
        conv = tf_layers.BatchNorm(conv, is_training, trainable = trainable,
                                   center=True, scale =True, activation_fn= tf.nn.relu,
                                   scope='bn0')
        conv = tf_layers.Conv2d(conv, depth, [3,3], 1, weight_initializer, bias_initializer,
                                weight_regulerization  = 'l2',
                                padding = 'SAME', activation_fn = None,
                                scope='conv1', trainable=trainable)
        conv = tf_layers.BatchNorm(conv, is_training, trainable = trainable,
                                   center=True, scale =True, activation_fn= tf.nn.relu,
                                   scope='bn1')
        
        if (stride > 1) | (x.get_shape().as_list()[-1] != depth):
            x = tf_layers.Conv2d(x, depth, [1,1], stride, weight_initializer, bias_initializer,
                                 weight_regulerization  = 'l2',
                                 padding = 'SAME', activation_fn = None,
                                 scope='conv2', trainable=trainable)
            x = tf_layers.BatchNorm(x, is_training, trainable = trainable,
                                    center=True, scale =True, activation_fn= tf.nn.relu,
                                    scope='bn2')

        x = x+conv
        x = activation_fn(x) if activation_fn != None else x
        return x