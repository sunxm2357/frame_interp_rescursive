#!/bin/bash

python src/trivial_test.py  \
    --name=kth_SA_P_F\
    --K=${1} \
    --F=${1} \
    --T=${2} \
    --data=KTH --dataroot=${3} --textroot=videolist/KTH/ --pick_mode=Slide \
    --image_size 128 \
