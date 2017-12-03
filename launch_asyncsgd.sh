#!/bin/bash

export TF_LOG_DIR="/home/ubuntu/tf/logs"
source /home/ubuntu/assign3_parta/uploaded-scripts/tfdefs.sh

# startserver.py has the specifications for the cluster.

terminate_cluster
start_cluster startserver.py

#rm -rf async_sgd_log.*
#rm -rf error_rates_async_sgd.txt

echo "Executing the distributed tensorflow job from asyncsgd.py"

#for i in 10
#do
START=$(date +%s)

nohup python asyncsgd.py --task_index=0 > async_sgd_log-0.out 2>&1&
sleep 2 # wait for variable to be initialized
nohup python asyncsgd.py --task_index=1 > async_sgd_log-1.out 2>&1&
nohup python asyncsgd.py --task_index=2 > async_sgd_log-2.out 2>&1&
nohup python asyncsgd.py --task_index=3 > async_sgd_log-3.out 2>&1&
nohup python asyncsgd.py --task_index=4 > async_sgd_log-4.out 2>&1&

#wait %1 %2 %3 %4 %5
END=$(date +%s)
DIFF=$(( $END - $START ))
echo "Program run time : $DIFF"
#done
# defined in tfdefs.sh to terminate the cluster
#tensorboard --logdir=$TF_LOG_DIR
#terminate_cluster
