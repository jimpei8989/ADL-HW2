from typing import Optional

NO_SPAN = -1


def pad_sequence(tokens, target_length, pad_token):
    return tokens + [pad_token] * (target_length - len(tokens))


def split_context_and_tokenize(
    question: str,
    context: str,
    start_index: int,
    end_index: int,
    tokenizer,
    tokenizer_max_length: int = 512,
    add_to_dict: Optional[dict] = {},
):
    # print(question, context, start_index)
    question_tokens = tokenizer.tokenize(question)
    maximum_length = tokenizer_max_length - 3 - len(question_tokens)

    raw = []

    fragment_start_index = 0
    while fragment_start_index < len(context):
        next_dot = context.find("。", fragment_start_index)
        fragment_end_index = min(
            fragment_start_index + maximum_length,
            next_dot + 1 if next_dot != -1 else len(context),
            len(context),
        )

        # print(fragment_start_index, fragment_end_index)

        text = context[fragment_start_index:fragment_end_index]
        if fragment_start_index <= start_index <= end_index <= fragment_end_index:
            _start, _end = start_index - fragment_start_index, end_index - fragment_start_index
            before, ans, after = map(
                tokenizer.tokenize, [text[:_start], text[_start:_end], text[_end:]]
            )
            tokens = before + ans + after
            label_start, label_end = len(before), len(before) + len(ans)
        else:
            tokens = tokenizer.tokenize(text)
            label_start, label_end = NO_SPAN, NO_SPAN

        raw.append(
            {
                "paragraph_text": text,
                "paragraph_tokens": tokens,
                "start_index": label_start,
                "end_index": label_end,
            }
        )
        fragment_start_index = fragment_end_index

    def merge_fragments(a, b):
        return {
            "paragraph_text": a["paragraph_text"] + b["paragraph_text"],
            "paragraph_tokens": a["paragraph_tokens"] + b["paragraph_tokens"],
            "start_index": (
                a["start_index"]
                if a["start_index"] != NO_SPAN
                else b["start_index"] + len(a["paragraph_tokens"])
                if b["start_index"] != NO_SPAN
                else NO_SPAN
            ),
            "end_index": (
                a["end_index"]
                if a["end_index"] != NO_SPAN
                else b["end_index"] + len(a["paragraph_tokens"])
                if b["end_index"] != NO_SPAN
                else NO_SPAN
            ),
        }

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

    return [
        d | {"question": question, "question_tokens": question_tokens} | add_to_dict
        for d in compressed
    ]


if __name__ == "__main__":
    from transformers import BertTokenizer

    question = "舍本和誰的數據能推算出連星的恆星的質量？"
    context = "在19世紀雙星觀測所獲得的成就使重要性也增加了。在1834年，白塞爾觀測到天狼星自行的變化，因而推測有一顆隱藏的伴星；愛德華·皮克林在1899年觀測開陽週期性分裂的光譜線時發現第一顆光譜雙星，週期是104天。天文學家斯特魯維和舍本·衛斯里·伯納姆仔細的觀察和收集了許多聯星的資料，使得可以從被確定的軌道要素推算出恆星的質量。第一個獲得解答的是1827年由菲利克斯·薩瓦里透過望遠鏡的觀測得到的聯星軌道。對恆星的科學研究在20世紀獲得快速的進展，相片成為天文學上很有價值的工具。卡爾·史瓦西發現經由比較視星等和攝影星等的差別，可以得到恆星的顏色和它的溫度。1921年，光電光度計的發展可以在不同的波長間隔上非常精密的測量星等。阿爾伯特·邁克生在虎克望遠鏡第一次使用干涉儀測量出恆星的直徑。"
    start, end = 108, 112
    print(context[start:end])
    fragments = split_context_and_tokenize(
        question, context, start, end, BertTokenizer.from_pretrained("bert-base-chinese")
    )
    print(fragments)
