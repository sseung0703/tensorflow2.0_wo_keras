import tensorflow as tf

def Flatten(X):
    sz = X.get_shape().as_list()
    S = 1
    for s in sz[1:]:
        S*=s
    return tf.reshape(X, [-1,S])
    
def Conv2d(X, depth, kernel, stride,  
           weight_initializer, bias_initializer,
           weight_regulerization  = None,
           bias_regulerization  = None,
           padding = 'SAME', activation_fn = tf.nn.relu,
           scope=None, trainable=False, reuse = False
           ):
    
    graph = tf.compat.v1.get_default_graph()
    with tf.name_scope(scope):
        B,H,W,D = X.get_shape().as_list()
        
        if reuse == False:
            weights = tf.Variable(weight_initializer(kernel + [D, depth], tf.float32), name = 'weights',
                                  trainable=trainable)
        else:
            weights = [v for v in graph.get_collection('variables') if v.name == graph.get_name_scope()+'/%s/weights:0'%scope][0]
        
        if trainable:
            if weight_regulerization is not None:
                if weight_regulerization is 'l2':
                    graph.add_to_collections('reg', tf.reduce_sum(tf.square(weights))/2)
                if weight_regulerization is 'l1':
                    graph.add_to_collections('reg', tf.reduce_sum(tf.abs(weights)))
        
        X = tf.nn.conv2d(X, weights, [1,stride,stride,1], padding, data_format='NHWC', name='convolution')
        if bias_initializer is not None:
            if reuse == False:
                biases = tf.Variable(bias_initializer([depth], tf.float32), name = 'biases',
                                     trainable = trainable)
            else:
                biases = [v for v in graph.get_collection('variables') if v.name == graph.get_name_scope()+'/%s/biases:0'%scope][0]
            X += biases
            if trainable:
                if bias_regulerization is not None:
                    if bias_regulerization is 'l2':
                        graph.add_to_collections('reg', tf.reduce_sum(tf.square(biases))/2)
                    if bias_regulerization is 'l1':
                        graph.add_to_collections('reg', tf.reduce_sum(tf.abs(biases)))
        
        if activation_fn is not None:
            X = activation_fn(X)
            
    return X

def Dense(X, depth,   
          weight_initializer, bias_initializer,
          weight_regulerization  = None,
          bias_regulerization  = None,
          activation_fn = tf.nn.relu,
          scope=None, trainable=False, reuse = False
          ):
    
    graph = tf.compat.v1.get_default_graph()
    with tf.name_scope(scope):
        B, D = X.get_shape().as_list()
        
        if reuse == False:
            weights = tf.Variable(weight_initializer([D, depth], tf.float32), name = 'weights',
                                  trainable=trainable)
        else:
            weights = [v for v in graph.get_collection('variables') if v.name == graph.get_name_scope()+'weights:0'][0]
        
        if trainable:
            if weight_regulerization is not None:
                if weight_regulerization is 'l2':
                    graph.add_to_collections('reg', tf.reduce_sum(tf.square(weights))/2)
                if weight_regulerization is 'l1':
                    graph.add_to_collections('reg', tf.reduce_sum(tf.abs(weights)))
        
        X = tf.matmul(X, weights)
        if bias_initializer is not None:
            if reuse == False:
                biases = tf.Variable(bias_initializer([depth], tf.float32), name = 'biases',
                                     trainable = trainable)
            else:
                biases = [v for v in graph.get_collection('variables') if v.name == graph.get_name_scope()+'biases:0'][0]
            X += biases
            if trainable:
                if bias_regulerization is not None:
                    if bias_regulerization is 'l2':
                        graph.add_to_collections('reg', tf.reduce_sum(tf.square(biases))/2)
                    if bias_regulerization is 'l1':
                        graph.add_to_collections('reg', tf.reduce_sum(tf.abs(biases)))
        
        if activation_fn is not None:
            X = activation_fn(X)
            
    return X


def BatchNorm(X,
              is_training,
              trainable = False,
              decay=0.999,
              center=True,
              scale=False,
              epsilon=0.001,
              activation_fn=None,
              param_initializers=None,
              param_regularizers=None,
              reuse=None,
              scope=None,
              ):
    graph = tf.compat.v1.get_default_graph()
    with tf.name_scope(scope):
        D, *rest = X.get_shape().as_list()[::-1]
        rest_dim = len(rest)
        
        if param_initializers is None:
            init_ones = tf.keras.initializers.Ones()([1]*rest_dim+[D], tf.float32)
            init_zeros = tf.keras.initializers.Zeros()([1]*rest_dim+[D], tf.float32)
            param_initializers = {'moving_mean' :     init_zeros,
                                  'moving_variance' : init_ones}
            if scale:
                param_initializers['gamma'] = init_ones
                
            if center:
                param_initializers['beta'] = init_zeros
            
        moving_mean = tf.Variable(param_initializers['moving_mean'], name = 'moving_mean', trainable=trainable)
        moving_variance  = tf.Variable(param_initializers['moving_variance'], name = 'moving_variance', trainable=trainable)
        graph.add_to_collection('BN_collection', moving_mean)
        graph.add_to_collection('BN_collection', moving_variance)
        
        def training_phase(X, moving_mean, moving_variance):
            mean, var= tf.nn.moments(X, list(range(len(rest))), keepdims=True)
            graph.add_to_collection('update_ops', moving_mean.assign(moving_mean - (1 - decay) * (moving_mean - mean)))
            graph.add_to_collection('update_ops', moving_variance.assign(moving_variance- (1 - decay) * (moving_variance - var)))
            return (X-mean)/tf.sqrt(var+epsilon)
        
        def inference_phase(X, moving_mean, moving_var):
            return (X-moving_mean)/tf.sqrt(moving_variance+epsilon)
        
        X_norm = tf.cond(is_training, lambda : training_phase( X, moving_mean, moving_variance),
                                      lambda : inference_phase(X, moving_mean, moving_variance))
        
        if scale:
            gamma = tf.Variable(param_initializers['gamma'], name = 'gamma',
                               trainable=trainable)
            graph.add_to_collection('BN_collection', gamma)
            X_norm *= gamma
            
        if center:
            beta = tf.Variable(param_initializers['beta'], name = 'beta',
                               trainable=trainable)
            graph.add_to_collection('BN_collection', beta)
            X_norm += beta

        if activation_fn is not None:
            X_norm = activation_fn(X_norm)
            
    return X_norm