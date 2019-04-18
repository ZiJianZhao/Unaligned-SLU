#!/bin/bash

task=slu
mode=stat

experiment=base-hd
data_root=../dstc2-slu/manual-model1/
save_model=model1-v1-3.pt

deviceid=1

epochs=50
batchs=20

optim=adam
lr=0.001
maxnorm=5

test_file=test
save_file=manual.json

python base/hd/main.py -task $task -mode $mode -experiment $experiment -data_root $data_root \
    -save_model $save_model -deviceid $deviceid \
    -epochs $epochs -batch_size $batchs \
    -optim $optim -lr $lr -max_norm $maxnorm \
    -test_file $test_file -save_file $save_file \
    -load_word_emb -load_class_emb -share_param
