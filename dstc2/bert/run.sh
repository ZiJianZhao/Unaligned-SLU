#!/bin/bash

mode=train
experiment=bert
data_root=../dstc2-slu/manual/
save_model=first-manual.pt

deviceid=1

bert_mode=first
hid_dim=768

epochs=50
batchs=20

optim=bert
lr=5e-5
maxnorm=5

test_json=test.json
save_decode=first-manual.json
chkpt=first-manual.pt

python dstc2/bert/main.py -mode $mode -experiment $experiment -data_root $data_root \
    -save_model $save_model -deviceid $deviceid \
    -bert_mode $bert_mode -hid_dim $hid_dim \
    -epochs $epochs -batch_size $batchs \
    -optim $optim -lr $lr -max_norm $maxnorm \
    -test_json $test_json -load_chkpt $chkpt -save_decode $save_decode
