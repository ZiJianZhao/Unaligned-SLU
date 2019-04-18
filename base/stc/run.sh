#!/bin/bash

mode=test
experiment=base-stc
data_root=../dstc2-slu/manual/
save_model=RNN2One.pt

deviceid=1

model_type=RNN2One

epochs=50
batchs=20

optim=adam
lr=0.001
maxnorm=5

test_json=test.json
save_decode=manual-RNN2One.json
chkpt=RNN2One.pt

python base/stc/main.py -mode $mode -experiment $experiment -data_root $data_root \
    -save_model $save_model -deviceid $deviceid \
    -model_type $model_type -load_emb \
    -epochs $epochs -batch_size $batchs \
    -optim $optim -lr $lr -max_norm $maxnorm \
    -test_json $test_json -load_chkpt $chkpt -save_decode $save_decode
