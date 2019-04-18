#!/bin/bash

task=slu
mode=train

experiment=dstc3-hd
data_root=../dstc3/manual/
memory_path=dstc2_seed.memory.pt
train_file=dstc2.train
valid_file=dstc2.valid

load_model=none
save_model=dstc2-pre-dropout.pt

deviceid=0

epochs=100
batchs=20

optim=adam
lr=0.001
maxnorm=5

test_file=test.json
save_file=tmp.json

python dstc3/hd-dropout/main.py -task $task -mode $mode -experiment $experiment -data_root $data_root \
    -memory_path $memory_path -train_file $train_file -valid_file $valid_file \
    -save_model $save_model -deviceid $deviceid -load_model $load_model \
    -epochs $epochs -batch_size $batchs \
    -optim $optim -lr $lr -max_norm $maxnorm \
    -test_file $test_file -save_file $save_file \
    -load_word_emb -load_class_emb -share_param
