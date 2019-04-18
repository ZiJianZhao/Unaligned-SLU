#!/bin/bash

task=slu
mode=test

experiment=base-hd-trans
data_root=../dstc2-slu/1best-live/
save_model=1best-live-2-100.pt

deviceid=0

num_layers=2
hid_dim=100

epochs=50
batchs=20

optim=adam
lr=0.001
maxnorm=5

test_file=test.json
save_file=1best-live.json

python base/hd-trans/main.py -task $task -mode $mode -experiment $experiment -data_root $data_root \
    -save_model $save_model -deviceid $deviceid -num_layers $num_layers -hid_dim $hid_dim \
    -epochs $epochs -batch_size $batchs \
    -optim $optim -lr $lr -max_norm $maxnorm \
    -test_file $test_file -save_file $save_file \
    -load_word_emb -load_class_emb -share_param
