# Homework 2 - Chinese Question Answering
> Applied Deep Learning (CSIE 5431)

## Shortcuts
- [Instruction slides (Google slides)](https://docs.google.com/presentation/d/1eonDCBNEqbvAEGKqPWt3Ew1JjVlBYXX45G2Hqs7c0Hk/edit)
- There's no Kaggle competition page this time

## Environment
- Python `3.9.4`
- Requirements: please refer to [requirements.txt](requirements.txt)
- Virtual environment using `pyenv`
- CPU: AMD Ryzen 7 3700X
- GPU: NVIDIA GeForce RTX 2070 Super

## Reproduce My Best Model

- Context Selection
    ```bash
    python3 src/context_selection.py \
        configs/context_selection/macbert_base.json \
        --do_train --gpu
    ```
    - Training time: about 50min / epoch

- Question Answering
    ```bash
    python3 src/question_answering.py \
        configs/question_answering/macbert_base.json \
        --do_train --gpu
    ```
    - Training time: about 20min / epoch
