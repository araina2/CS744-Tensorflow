import tensorflow as tf
import os
import time

tf.logging.set_verbosity(tf.logging.DEBUG)
num_features = 33762578

tf.app.flags.DEFINE_integer("task_index", 0, "Index of the worker task")
FLAGS = tf.app.flags.FLAGS

train_file_loc = [[
          "/home/ubuntu/criteo-tfr-tiny/tfrecords00",
          "/home/ubuntu/criteo-tfr-tiny/tfrecords01",
          "/home/ubuntu/criteo-tfr-tiny/tfrecords02",
          "/home/ubuntu/criteo-tfr-tiny/tfrecords03",
          "/home/ubuntu/criteo-tfr-tiny/tfrecords04"
           ],
           [ 
           "/home/ubuntu/criteo-tfr-tiny/tfrecords05",
           "/home/ubuntu/criteo-tfr-tiny/tfrecords06",
           "/home/ubuntu/criteo-tfr-tiny/tfrecords07",
           "/home/ubuntu/criteo-tfr-tiny/tfrecords08",
           "/home/ubuntu/criteo-tfr-tiny/tfrecords09"
           ],
           [
           "/home/ubuntu/criteo-tfr-tiny/tfrecords10",
           "/home/ubuntu/criteo-tfr-tiny/tfrecords11",
           "/home/ubuntu/criteo-tfr-tiny/tfrecords12",
           "/home/ubuntu/criteo-tfr-tiny/tfrecords13",
           "/home/ubuntu/criteo-tfr-tiny/tfrecords14"
          ],
          [
           "/home/ubuntu/criteo-tfr-tiny/tfrecords15",
           "/home/ubuntu/criteo-tfr-tiny/tfrecords16",
           "/home/ubuntu/criteo-tfr-tiny/tfrecords17",
           "/home/ubuntu/criteo-tfr-tiny/tfrecords18",
           "/home/ubuntu/criteo-tfr-tiny/tfrecords19"
          ],
          [
           "/home/ubuntu/criteo-tfr-tiny/tfrecords20",
           "/home/ubuntu/criteo-tfr-tiny/tfrecords21"
          ]]

test_file_loc = "/home/ubuntu/criteo-tfr-tiny/tfrecords22"


eta = 0.01

g = tf.Graph()
iterations = 20000000
#iterations = 2

error_rate = 0

def test_model() :
	with tf.device("/job:worker/task:%d" % 0):
		filename_queue = tf.train.string_input_producer([test_file_loc],num_epochs=None)
		reader = tf.TFRecordReader()
		_, serialized_data = reader.read(filename_queue)
		features = tf.parse_single_example(serialized_data,
												   features={
														'label': tf.FixedLenFeature([1], dtype=tf.int64),
														'index' : tf.VarLenFeature(dtype=tf.int64),
														'value' : tf.VarLenFeature(dtype=tf.float32),
												   }
												  )

		label = features['label']
		index = features['index']
		value = features['value']


		dense_feature = tf.sparse_to_dense(tf.sparse_tensor_to_dense(index),[num_features,],tf.sparse_tensor_to_dense(value))

		prediction = tf.reduce_sum(tf.mul(w,dense_feature))
		sign = tf.sign(prediction)
		sign = tf.cast(sign,tf.int64)
		temp = tf.sub(sign,label)
		temp = tf.abs(temp)
		temp = tf.cast(temp, tf.float32)
		temp = tf.cast(temp, tf.float32)
		error = tf.mul(0.5,temp)
	return error


def train_model(feature, label, device) :
	with tf.device("/job:worker/task:%d" % device):
		sparse_weights = tf.gather(w, feature['index'].values)
		wdotx = tf.reduce_sum(tf.mul(sparse_weights,feature['value'].values))
		y = tf.cast(label, tf.float32)
		ywdotx = tf.mul(y, wdotx)
		mul_factor = tf.mul(tf.sigmoid(ywdotx)-1,feature['value'].values)

		gradient = tf.mul(y, mul_factor)
		gradient_update = tf.mul((-1.0*eta), gradient)
		sparse_gradient_update = tf.SparseTensor(shape=[num_features], indices=[feature['index'].values], values=gradient_update)
		return sparse_gradient_update

### Graph execution starts

with g.as_default():
	with tf.device("/job:worker/task:0"):
		w = tf.Variable(tf.ones([num_features]), name="model")

	# Below code is for training one sample on 5 VMs
	#for i in range(0,5):
	with tf.device("/job:worker/task:%d" % FLAGS.task_index):
		filename_queue = tf.train.string_input_producer(train_file_loc[FLAGS.task_index], num_epochs=None)
		reader = tf.TFRecordReader()
		_, serialized_data = reader.read(filename_queue)
		features = tf.parse_single_example(serialized_data,
												   features={
														'label': tf.FixedLenFeature([1], dtype=tf.int64),
														'index' : tf.VarLenFeature(dtype=tf.int64),
														'value' : tf.VarLenFeature(dtype=tf.float32),
												   }
												  )

		label = features['label']
		index = features['index']
		value = features['value']

		sparse_gradient_update = train_model(features,label,FLAGS.task_index)
			#w = tf.scatter_add(w,  tf.reshape(sparse_gradient_update.indices, [-1]), tf.reshape(sparse_gradient_update.values, [-1]))


	with tf.device("/job:worker/task:0"):
		#dense_gradient_update = tf.sparse_to_dense(tf.transpose(sparse_gradient_update.indices),[num_features],sparse_gradient_update.values)
		w = tf.scatter_add(w,  tf.reshape(sparse_gradient_update.indices, [-1]), tf.reshape(sparse_gradient_update.values, [-1]))
		#new_w = tf.sparse_add(w, sparse_gradient_update)
		test_error = test_model()

	#config = tf.ConfigProto(log_device_placement=True)
	with tf.Session("grpc://vm-18-%d:2222" % (FLAGS.task_index+1)) as sess:
		coord = tf.train.Coordinator()
		if FLAGS.task_index == 0:
			sess.run(tf.initialize_all_variables())
			sess.run(tf.initialize_local_variables())
		#tf.train.start_queue_runners(sess=sess)
		tf.train.start_queue_runners(sess=sess, coord=coord)

		try:
			count = 0
			while not coord.should_stop():
			#for i in range(0, iterations):
				if count >= iterations:
					break
				if(count%100 == 0):
					test_num = 0
					errors = []
					try:
						while test_num < 10:
							result = sess.run(test_error)
							errors.append(result)
							#print "Printing errors vector"
							#print errors
							test_num = test_num +1
					except tf.errors.OutOfRangeError :
						print "Detected EOF in the test file"
					count_errors = 0
					for error in errors :
						if error == 1 :
							count_errors+=1
					if len(errors) > 0:
						error_rate = (count_errors*1.0)/len(errors)
					#print ("Length of the errors array is:" +str(len(errors)))
					#print ("Value of count_errors is: " +str(count_errors))
					print "Test Error Rate : "+str(error_rate)
					#print "Printing errors vector"
					#print errors
					with open("error_rates_async_sgd.txt", "a+") as error_file :
						error_file.write("\nError Rate after "+str(count)+" iteration "+str(error_rate))
					count+=1
					print "Number of examples read : "+str(count)

				#start = time.time()
				#sess.run(sparse_gradient_update)
				#end = time.time()
				#time_taken  = end - start
				#print "Time spent in getting the gradient updates: "+str(time_taken)
				#start = time.time()
				#w.eval()
				#end = time.time()
				#time_taken  = end - start
			 	#print "Time spent in getting the updated weight is: "+str(time_taken)
				#print "The new w is :"
				#print w.eval()
				
		except tf.errors.OutOfRangeError:
			print("Training is done.\n")
		finally:
			coord.request_stop()
		coord.join()

		#tf.train.SummaryWriter("%s/async_sgd" % (os.environ.get("TF_LOG_DIR")), sess.graph)
		sess.close()

