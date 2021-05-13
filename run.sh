#! /usr/bin/env bash

CONTEXT_JSON=${1}
TEST_JSON=${2}
PREDICT_JSON=${3}


python3 src/predict.py \
    configs/context_selection/bert_base.json \
    configs/question_answering/bert_base.json \
    --context_json ${CONTEXT_JSON} \
    --test_json ${TEST_JSON} \
    --predict_json ${PREDICT_JSON} \
    --gpu
