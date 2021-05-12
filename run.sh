#! /usr/bin/env bash

CONTEXT_JSON=${1}
TEST_JSON=${2}
PREDICT_CSV=${3}


python3 src/predict.py \
    --context_json ${CONTEXT_JSON} \
    --test_json ${TEST_JSON} \
    --predict_csv ${PREDICT_CSV}
