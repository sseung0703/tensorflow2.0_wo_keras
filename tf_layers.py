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