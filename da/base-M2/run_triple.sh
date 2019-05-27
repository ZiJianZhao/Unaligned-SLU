#!/bin/bash

mode=gen

experiment=dstc3-da
data_root=../dstc3-st/manual-da/
memory_path=del/memory.pt
train_file=dstc3.train
valid_file=dstc3.valid

load_model=del.dstc2train.M2.pt
save_model=del.dstc2train.3seed.M2.pt

deviceid=1

epochs=50
batchs=20

optim=adam
lr=0.001
maxnorm=5

test_file=test
class_file=trp/e3
save_file=e3.del.1best.train
nbest=1

python da/base-M2/main.py -mode $mode -experiment $experiment -data_root $data_root \
    -memory_path $memory_path -train_file $train_file -valid_file $valid_file \
    -save_model $save_model -deviceid $deviceid \
    -epochs $epochs -batch_size $batchs \
    -optim $optim -lr $lr -max_norm $maxnorm \
    -test_file $test_file -save_file $save_file -nbest $nbest -class_file $class_file \
    -load_word_emb -load_model $load_model 
#-enlarge_word_vocab
