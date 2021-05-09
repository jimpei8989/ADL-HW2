def split_context_and_tokenize(
    context: str, maximum_length: int, start_index: int, end_index: int, tokenizer
):
    fragments = []

    fragment_start_index = 0
    while fragment_start_index < len(context):
        fragment_end_index = min(
            fragment_start_index + maximum_length,
            context.find("。", fragment_start_index) + 1,
            len(context),
        )

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
            label_start, label_end = -1, -1

        fragments.append(
            {
                "text": text,
                "tokens": tokens,
                "start_index": label_start,
                "end_index": label_end,
            }
        )
        fragment_start_index = fragment_end_index

    def merge_fragments(a, b):
        return {
            "text": a["text"] + b["text"],
            "tokens": a["tokens"] + b["tokens"],
            "start_index": (
                a["start_index"]
                if a["start_index"] != -1
                else b["start_index"] + len(a["tokens"])
                if b["start_index"] != -1
                else -1
            ),
            "end_index": (
                a["end_index"]
                if a["end_index"] != -1
                else b["end_index"] + len(a["tokens"])
                if b["end_index"] != -1
                else -1
            ),
        }

    compressed_fragments = []
    for fragment in fragments:
        if (
            len(compressed_fragments) == 0
            or len(compressed_fragments[-1]["tokens"]) + len(fragment["tokens"]) > maximum_length
        ):
            compressed_fragments.append(fragment)
        else:
            compressed_fragments[-1] = merge_fragments(compressed_fragments[-1], fragment)

    return compressed_fragments


if __name__ == "__main__":
    from transformers import BertTokenizer

    context = "在19世紀雙星觀測所獲得的成就使重要性也增加了。在1834年，白塞爾觀測到天狼星自行的變化，因而推測有一顆隱藏的伴星；愛德華·皮克林在1899年觀測開陽週期性分裂的光譜線時發現第一顆光譜雙星，週期是104天。天文學家斯特魯維和舍本·衛斯里·伯納姆仔細的觀察和收集了許多聯星的資料，使得可以從被確定的軌道要素推算出恆星的質量。第一個獲得解答的是1827年由菲利克斯·薩瓦里透過望遠鏡的觀測得到的聯星軌道。對恆星的科學研究在20世紀獲得快速的進展，相片成為天文學上很有價值的工具。卡爾·史瓦西發現經由比較視星等和攝影星等的差別，可以得到恆星的顏色和它的溫度。1921年，光電光度計的發展可以在不同的波長間隔上非常精密的測量星等。阿爾伯特·邁克生在虎克望遠鏡第一次使用干涉儀測量出恆星的直徑。"
    start, end = 108, 112
    print(context[start:end])
    fragments = split_context_and_tokenize(
        context, 372, start, end, BertTokenizer.from_pretrained("bert-base-chinese")
    )
    print(fragments)
