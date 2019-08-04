#!/bin/bash


data_dir=../dstc3-st/s2s/s2s-form-1/
train_file=dstc2.train
valid_file=dstc2.valid
vocab_file=dstc2.vocab.pt

experiment=s2s/s2s-form-1/
save_model=dstc2.best.pt

gpuid=0
epochs=20
batch_size=20

python dual/pointer/train.py \
    -data_dir $data_dir -train_file $train_file -valid_file $valid_file  -vocab_file $vocab_file \
    -experiment $experiment -save_model $save_model \
    -gpuid $gpuid -epochs $epochs -batch_size $batch_size
