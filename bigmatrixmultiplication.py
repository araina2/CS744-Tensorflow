"""
TF client program for big matrix multiplication
"""

import tensorflow as tf

# create a empty graph
g = tf.Graph()

# make the graph we created as the default graph. All the variables and
# operators we create will be added to this graph
with g.as_default():

    tf.logging.set_verbosity(tf.logging.DEBUG)

    # this sets the random seed for operations on the current graph
    tf.set_random_seed(1)

    M = 100000
    split = 100
  
    matrices = {} # a container to hold the operators we just created
    cnt = 0
    for i in range(0, split):
        for j in range (0, split):
            task_num = cnt % 5
            cnt = cnt + 1
            with tf.device("/job:worker/task:%d" % task_num):
                matrix_name = "matrix-"+str(i)+str(j)
            		# create a operator that generates a random_normal tensor of the
            		# specified dimensions. By placing this statement here, we ensure
            		# that this operator is executed on "/job:worker/task:task_num". Note that
                # are creating split*split matrices of size M/split.
                matrices[matrix_name] = tf.random_normal([M/split, M/split], name=matrix_name)

    # container to hold operators that calculate the traces of individual matrices.
    intermediate_traces = {}
    
    cnt = 0
    for i in range(0, split):
        for j in range(0, split):
            task_num = cnt % 5
            cnt = cnt + 1
            with tf.device("/job:worker/task:%d" % task_num):
                A = matrices["matrix-"+str(i)+str(j)]
                B = matrices["matrix-"+str(j)+str(i)]
            		# tf.trace() will create an operator that takes a single matrix
                intermediate_traces["matrix-"+str(i)+str(j)] = tf.trace(tf.matmul(A, B))
    
    # sum all the traces
    with tf.device("/job:worker/task:0"):
        retval = tf.add_n(intermediate_traces.values())

    config = tf.ConfigProto(log_device_placement=True)
    with tf.Session("grpc://vm-18-1:2222", config=config) as sess:
        result = sess.run(retval)
        sess.close()
        print "The trace of A*A is : %f" % result
