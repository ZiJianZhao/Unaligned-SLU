#!/bin/bash

mode=train

experiment=dstc3-da
data_root=../dstc3-st/manual-da/tmp/
memory_path=memory.pt
train_file=dstc3.1000.train
valid_file=dstc3.valid

#load_model=tmp.dstc2train.3seed.M2.auto.pt
load_model=tmp.dstc2train.M2.pt
save_model=tmp.dstc2train.1000.M2.pt

deviceid=1

epochs=50
batchs=20

optim=adam
lr=0.001
maxnorm=5

test_file=new.value.test.templete
class_file=dstc3.test.triples.json
save_file=decodes/tmp.auto.dstc2train.3seed.M2.on.test.1
nbest=1

python da/templete-M2/main.py -mode $mode -experiment $experiment -data_root $data_root \
    -memory_path $memory_path -train_file $train_file -valid_file $valid_file \
    -save_model $save_model -deviceid $deviceid \
    -epochs $epochs -batch_size $batchs \
    -optim $optim -lr $lr -max_norm $maxnorm \
    -test_file $test_file -save_file $save_file -nbest $nbest -class_file $class_file \
    -load_word_emb -load_model $load_model
