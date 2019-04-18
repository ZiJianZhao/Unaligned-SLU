#!/bin/bash

task=slu
mode=test

experiment=dstc3-st
data_root=../dstc3-st/manual-da/genates/
memory_path=dstc2-3train-on-test-6.memory.pt
train_file=dstc2-3train-on-test-6
valid_file=valid

load_model=none
save_model=dstc2-3train-on-test-6.pt

deviceid=0

epochs=50
batchs=20

optim=adam
lr=0.001
maxnorm=5

test_file=test.json
save_file=tmp.json

python dstc3/hd-ontology/main.py -task $task -mode $mode -experiment $experiment -data_root $data_root \
    -memory_path $memory_path -train_file $train_file -valid_file $valid_file \
    -save_model $save_model -deviceid $deviceid -load_model $load_model \
    -epochs $epochs -batch_size $batchs \
    -optim $optim -lr $lr -max_norm $maxnorm \
    -test_file $test_file -save_file $save_file \
    -load_word_emb -load_class_emb -share_param
