#!/bin/bash

task=slu
mode=test

experiment=wcn
data_root=../dstc2-slu/wcn-partial/
save_model=wcn-partial-mwsr.pt

deviceid=1

wcn_emb_type=mwsr

epochs=50
batchs=20

optim=adam
lr=0.001
maxnorm=5

test_file=test.json
save_file=wcn-partial-mwsr.json

python dstc2/wcn/main.py -task $task -mode $mode -experiment $experiment -data_root $data_root \
    -save_model $save_model -deviceid $deviceid \
    -epochs $epochs -batch_size $batchs \
    -optim $optim -lr $lr -max_norm $maxnorm \
    -test_file $test_file -save_file $save_file \
    -load_word_emb -load_class_emb -share_param -wcn_emb_type $wcn_emb_type
