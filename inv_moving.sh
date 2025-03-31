#!/bin/bash

cd $(dirname $0)

source activate devito4.8.2
cur_dateTime=$(date "+%Y%m%d-%H%M%S")
if (($#==0))
then 
    nohup python -u inv_moving.py 1>./logs/inv_moving_$cur_dateTime.out 2>./logs/inv_moving_$cur_dateTime.err &
elif (($#==1))
then
    jsonfile=$1
    if [ ! -f $jsonfile ]; then 
        echo "JSON file does not exist!"
    else
        nohup python -u inv_moving.py $1 1>./logs/inv_moving_$cur_dateTime.out 2>./logs/inv_moving_$cur_dateTime.err &
    fi
else 
    echo "Invalid number of argments!"
    exit 1
fi