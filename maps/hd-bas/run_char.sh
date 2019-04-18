#!/bin/bash

task=slu
mode=stat

experiment=maps
data_root=../maps/char/
save_model=char.pt

deviceid=0

epochs=50
batchs=80

optim=adam
lr=0.001
maxnorm=5

test_file=test
save_file=manual.json

python maps/hd/main.py -task $task -mode $mode -experiment $experiment -data_root $data_root \
    -save_model $save_model -deviceid $deviceid \
    -epochs $epochs -batch_size $batchs \
    -optim $optim -lr $lr -max_norm $maxnorm \
    -test_file $test_file -save_file $save_file \
    -share_param -load_word_emb
