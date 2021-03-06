# Unaligned-SLU

This repository contains codes for unaligned SLU. 
Given an utterance (manual, 1best, nbest, word confusion networks), 
the labels (act-slot-value triples) are predicted.

## xslu
The [`xlsu` folder](./xslu) contains general and common codes for unaligned SLU.
* utils
* modules
* optimizers

## text
The [`text` folder](./text) contains codes for text preprocessing:
* dstc2 preprocessing, rearange the json files to visible text files
* extract features, word2idx, class2idx and so on, then save them for future uses.

## base
The [`base` folder](./base) contains two basic different methods for unaligned SLU:
* [`stc`](./base/stc): (semantic tuple classifier), treat the whole act-slot-value triple as a label, 
    and define the task as a multi-label classification task.
* [`hd`](./base/hd): (hierarchical decoding model), use different but interconnected modules to predict
    the act-slot-value triples hierarchically. Typically, the predictor for value is a 
    Seq2Seq-Attention model with Pointer network.

And the example codes are based on the dataset provide by DSTC2. These codes can be used as 
templates when coding for some specific tasks, e.g., word confusion networks application, 
transformer experiment, and so on. For each new task or setting, we have a copy of the 
corresponding base method, and modify the code to fit the setting. Therefore, there are lots of 
redundancy in the codes, but the benefit is that the codes don't affect each other.

## dstc2
The [`dstc2` folder](./dstc2) contains codes of different models for unaligned SLU task in DSTC2:
* [`bert`](./dstc2/bert): Utilization of the paper [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805),
    with only stc method just for a try.
* [`wcn`](./dstc2/wcn): Research of word confusion networks to enhance robustness to ASR errors.

And the codes in this folder is to research for:
* Variant models for improving performance and generalization quality
* Robustness to ASR-errors

## dstc3
The [`dstc3` folder](./dstc3) contains codes of different models for unaligned SLU task in DSTC3:
* [`da`](./dstc3/da): Data augmentation.
* [`hd`](./dstc3/hd): hierarchical decoding model on dstc3 data.


## maps
The utilization of hierarchical decoding model (hd) in real datas (Chinese maps). The difference of the data from dstc2 is that 
the length of values is much longer which can be used to test the effectiveness of pointer network in value decoder. Also, we try
different vocabularies in the decoder input and output ends.
* ['hd-bas'](./maps/hd-bas): the input and output ends share the same basic vocabulary constructed from the words in utterances in training file.
* ['hd-spe'](./maps/hd-spe): the input end uses the basic (or pre-trained) vocabulary while the output end uses a special and small vocabulary including
                            the enumerable values which can not be copied from the input utterance.
