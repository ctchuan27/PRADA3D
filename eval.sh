#!/bin/bash

python eval.py -s ./default_dataset/m3c_processed -m ./output/m3c_processed_v2 --train_stage 1 --epoch 300
python eval.py -s ./default_dataset/m4c_processed -m ./output/m4c_processed_v2 --train_stage 1 --epoch 300
python eval.py -s ./default_dataset/f3c_processed -m ./output/f3c_processed_v2 --train_stage 1 --epoch 300
python eval.py -s ./default_dataset/f4c_processed -m ./output/f4c_processed_v2 --train_stage 1 --epoch 300

#python eval.py -s ./default_dataset/m3c_processed -m ./output/m3c_processed --train_stage 1 --epoch 200
#python eval.py -s ./default_dataset/m4c_processed -m ./output/m4c_processed --train_stage 1 --epoch 200
#python eval.py -s ./default_dataset/f3c_processed -m ./output/f3c_processed --train_stage 1 --epoch 200
#python eval.py -s ./default_dataset/f4c_processed -m ./output/f4c_processed --train_stage 1 --epoch 200

