from typing import Optional
from torch.utils.data._utils.collate import default_collate
from torch.nn.utils.rnn import pad_sequence


def pack(data):
    keys = list(data.keys())
    assert all(len(data[k]) == len(data[keys[0]]) for k in keys[1:])
    return [{k: data[k][i] for k in data.keys()} for i in range(len(data[keys[0]]))]


def unpack(data):
    pass


def create_mini_batch(batchs, pad_keys={}, padding_value=0):
    return {
        k: (
            pad_sequence([d[k] for d in batchs], batch_first=True, padding_value=padding_value)
            if k in pad_keys
            else [d[k] for d in batchs]
            if isinstance(batchs[0][k], list)
            else default_collate([d[k] for d in batchs])
        )
        for k in batchs[0].keys()
    }


def split_context_and_tokenize(
    question: str,
    context: str,
    tokenizer,
    answer: Optional[str] = None,
    start_index: Optional[int] = None,
    end_index: Optional[int] = None,
    add_to_dict: Optional[dict] = {},
):
    question_tokens = tokenizer.tokenize(question)
    # maximum_length = tokenizer.max_len_sentences_pair - len(question_tokens)
    maximum_length = 509 - len(question_tokens)

    raw = []

    fragment_start_index = 0
    while fragment_start_index < len(context):
        next_dot = context.find("。", fragment_start_index)
        fragment_end_index = min(
            fragment_start_index + maximum_length,
            next_dot + 1 if next_dot != -1 else len(context),
            len(context),
        )

        text = context[fragment_start_index:fragment_end_index]
        tmp = {"paragraph_text": text}
        if (
            answer is not None
            and fragment_start_index <= start_index <= end_index <= fragment_end_index
        ):
            _start, _end = start_index - fragment_start_index, end_index - fragment_start_index
            before, ans, after = map(
                tokenizer.tokenize, [text[:_start], text[_start:_end], text[_end:]]
            )
            tmp.update(
                {
                    "paragraph_tokens": before + ans + after,
                    "has_answer": True,
                    "start_index": len(before),
                    "end_index": len(before) + len(ans),
                }
            )
        else:
            tmp.update({"paragraph_tokens": tokenizer.tokenize(text), "has_answer": False})

        raw.append(tmp)
        fragment_start_index = fragment_end_index

    def merge_fragments(a, b):
        ret = {
            "paragraph_text": a["paragraph_text"] + b["paragraph_text"],
            "paragraph_tokens": a["paragraph_tokens"] + b["paragraph_tokens"],
            "has_answer": False,
        }

        if a["has_answer"]:
            ret["has_answer"] = True
            ret["start_index"] = a["start_index"]
            ret["end_index"] = a["end_index"]
        elif b["has_answer"]:
            ret["has_answer"] = True
            ret["start_index"] = b["start_index"] + len(a["paragraph_tokens"])
            ret["end_index"] = b["end_index"] + len(a["paragraph_tokens"])

        return ret

    compressed = []
    for fragment in raw:
        if (
            len(compressed) == 0
            or len(compressed[-1]["paragraph_tokens"]) + len(fragment["paragraph_tokens"])
            > maximum_length
        ):
            compressed.append(fragment)
        else:
            compressed[-1] = merge_fragments(compressed[-1], fragment)

    def finalize(d):
        ret = d | {"question": question, "question_tokens": question_tokens} | add_to_dict
        if ret["has_answer"]:
            ret["answer_text"] = answer
        return ret

    return [finalize(d) for d in compressed]


if __name__ == "__main__":
    from transformers import BertTokenizer

    question = "舍本和誰的數據能推算出連星的恆星的質量？"
    context = "在19世紀雙星觀測所獲得的成就使重要性也增加了。在1834年，白塞爾觀測到天狼星自行的變化，因而推測有一顆隱藏的伴星；愛德華·皮克林在1899年觀測開陽週期性分裂的光譜線時發現第一顆光譜雙星，週期是104天。天文學家斯特魯維和舍本·衛斯里·伯納姆仔細的觀察和收集了許多聯星的資料，使得可以從被確定的軌道要素推算出恆星的質量。第一個獲得解答的是1827年由菲利克斯·薩瓦里透過望遠鏡的觀測得到的聯星軌道。對恆星的科學研究在20世紀獲得快速的進展，相片成為天文學上很有價值的工具。卡爾·史瓦西發現經由比較視星等和攝影星等的差別，可以得到恆星的顏色和它的溫度。1921年，光電光度計的發展可以在不同的波長間隔上非常精密的測量星等。阿爾伯特·邁克生在虎克望遠鏡第一次使用干涉儀測量出恆星的直徑。"  # noqa: E501
    start, end = 108, 112
    print(context[start:end])

    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    fragments = split_context_and_tokenize(
        question, context, tokenizer, context[start:end], start, end
    )

    for i, frag in enumerate(fragments):
        if frag["has_answer"]:
            answer = "".join(frag["paragraph_tokens"][frag["start_index"] : frag["end_index"]])
            print(f"Fragment {i} has answer {answer} -- {frag['answer_text']}")
