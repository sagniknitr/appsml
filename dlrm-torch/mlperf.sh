#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#check if extra argument is passed to the test
if [[ $# == 1 ]]; then
    dlrm_extra_option=$1
else
    dlrm_extra_option=""
fi
#echo $dlrm_extra_option

cpu=1
gpu=0
pt=1
c2=0

ncores=1 #12 #6
nsockets="0"


numa_cmd="" #run on one socket, without HT
dlrm_pt_bin="python dlrm_s_pytorch.py"
dlrm_c2_bin="python dlrm_s_caffe2.py"

data=random #synthetic
print_freq=100
rand_seed=727

c2_net="async_scheduling"

#Model param
mb_size=2048 #1024 #512 #256
nbatches=1 #500 #100
bot_mlp="13-512-256-128"
top_mlp="479-1024-1024-512-256-1"
emb_size=128
nindices=100
emb="1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1"
interaction="dot"
tnworkers=0
tmb_size=16384

#_args="--mini-batch-size="${mb_size}\
_args=" --num-batches="${nbatches}\
" --data-generation="${data}\
" --arch-mlp-bot="${bot_mlp}\
" --arch-mlp-top="${top_mlp}\
" --arch-sparse-feature-size="${emb_size}\
" --arch-embedding-size="${emb}\
" --num-indices-per-lookup="${nindices}\
" --arch-interaction-op="${interaction}\
" --numpy-rand-seed="${rand_seed}\
" --print-freq="${print_freq}\
" --print-time"\
" --enable-profiling "

c2_args=" --caffe2-net-type="${c2_net}


cmd="$dlrm_pt_bin --mini-batch-size=$mb_size --test-mini-batch-size=$tmb_size --test-num-workers=$tnworkers $_args $dlrm_extra_option"

set -x
eval $cmd
