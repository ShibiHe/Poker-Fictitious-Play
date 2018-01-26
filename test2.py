import tensorflow as tf

with tf.variable_scope('a') as scope:
    tf.get_variable_scope().reuse_variables()
    print scope.reuse
    with tf.variable_scope('b') as scope2:
        print scope2.reuse
