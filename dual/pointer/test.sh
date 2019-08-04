#!/bin/bash


test_file=../dstc3-st/s2s/ori/dstc2.test.json
vocab_file=../dstc3-st/s2s/s2s-form-1/dstc2.vocab.pt

experiment=s2s/s2s-form-1/
chkpt=dstc2.best.pt

gpuid=0

python dual/pointer/test.py -test_file $test_file -vocab_file $vocab_file \
    -experiment $experiment -chkpt $chkpt \
    -gpuid $gpuid
