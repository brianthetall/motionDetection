#!/bin/bash

export PYTHONPATH=$PYTHONPATH:/usr/local/lib/python3.9/site-packages/

. bin/activate

while [[ 1 ]]
do
    /usr/home/user/motionDetection/bin/python /home/user/motionDetection/motionDetect.py rtsp://admin:admin@172.172.172.32/11
done
