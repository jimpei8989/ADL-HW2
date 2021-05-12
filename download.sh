#! /usr/bin/env bash

BASE_URL="wjpei.csie.org:29102"

# 1 - create the directories if they do not exist
mkdir -p checkpoints/content_selection/bert_base/
mkdir -p checkpoints/question_answering/bert_base/

# 2 - download the intent model
CONTEXT_MODEL_URL="${BASE_URL}/content_selection/model_weights.pt"
CONTEXT_MODEL_PATH="checkpoints/content_selection/bert_base/model_weights.pt"
CONTEXT_SHASUM="c8c4e1f6ea0ec1f4dc0cf4c3f91ec82864db78fd"

if [[ -f ${CONTEXT_MODEL_PATH} ]]; then
    echo "${CONTEXT_MODEL_PATH} already exists, skip downloading"
else
    wget ${CONTEXT_MODEL_URL} -O ${CONTEXT_MODEL_PATH}
fi

# 2 - download the slot model
QA_MODEL_URL="${BASE_URL}/question_answering/model_weights.pt"
QA_MODEL_PATH="checkpoints/question_answering/bert_base/model_weights.pt"
QA_SHASUM="d4c3e14136d1eb26ca61f968a18a5c94077e1163"

if [[ -f ${QA_MODEL_PATH} ]]; then
    echo "${QA_MODEL_PATH} already exists, skip downloading"
else
    wget ${QA_MODEL_URL} -O ${QA_MODEL_PATH}
fi