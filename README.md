# CS744-Tensorflow #

## Big matrix multiplication ##
The source code is in bigmatrixmultiplication.py. In the code, we split the
bigger matrix into smaller matrices and the intermediate traces of the smaller
matrices are evalated and then added up to get the trace of the bigger matrix.

### Running instructions ###
./bigmatrixmultiplication.sh

## Synchronous SGD ##
The source code is in synchronoussgd.py. In the code, we first compute the
gradients on five VMs and then perform updates to the weight matrix making sure
that we wait for all the VMs to compute the gradients and then update the
weight matrix. vm-18-1 is responsible for updating the weights. The error rates
will be logged in error_rates_sync_sgd.txt

### Running instructions ###
./launch_synchronoussgd.sh

## Asynchronous SGD ##
The source code is in asynchronoussgd.py. In the code, we again compute the
gradients on five VMs but don't wait for gradient computation from all the VMs
to update the weight matrix. Here, we launch five tasks and each task computes
its own gradient and in vm-18-1, we update the weight matrix with the last
computed gradient from any of the VMs. The error rates will be logged in
error_rates_async_sgd.txt

### Running instructions ###
./launch_asyncsgd.sh 


