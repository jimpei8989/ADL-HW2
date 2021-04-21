from tqdm.contrib.concurrent import process_map

from utils.tqdmm import TQDMM_KWARGS


def concurrent_tokenize(tokenizer, texts):
    tokenized = process_map(
        tokenizer.tokenize, texts, max_workers=8, chunksize=512, **TQDMM_KWARGS
    )
    return tokenized
