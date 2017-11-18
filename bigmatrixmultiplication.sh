#!/bin/bash

# tfdefs.sh has helper function to start process on all VMs
# it contains definition for start_cluster and terminate_cluster
source tfdefs.sh

# startserver.py has the specifications for the cluster.
start_cluster startserver.py

echo "Executing the distributed tensorflow job from bigmatrixmultiplication.py"
# bigmatrixmultiplication.py is a client that can run jobs on the cluster.
python bigmatrixmultiplication.py

# defined in tfdefs.sh to terminate the cluster
terminate_cluster
