import argparse

from jamgpt.tokenizer.bpe import BPETokenizer
from jamgpt.dataloader import parquets_iter_batched


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate tokenizers")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/Volumes/StorageT4/data/fineweb-edu-parquet-shards/sample-100BT",
        help="Data directory for parquet files",
    )
    parser.add_argument(
        "--tokenizers",
        nargs="+",
        choices=["gpt2", "gpt4", "custom"],
        default=["gpt2", "gpt4", "custom"],
        help="Tokenizers to evaluate (default: all)",
    )
    parser.add_argument(
        "--custom_tokenizer_model",
        type=str,
        default=None,
        help="Path to custom tokenizer model directory (required if 'custom' is in --tokenizers)",
    )
    parser.add_argument(
        "--skip_fwe",
        action="store_true",
        help="Skip FineWeb-Edu train/val data evaluation",
    )
    return parser.parse_args()


def get_test_texts():
    """Return a dictionary of test texts for evaluation."""
    # Random text I got from a random website this morning
    news_text = r"""
(Washington, D.C., July 9, 2025)- Yesterday, Mexico's National Service of Agro-Alimentary Health, Safety, and Quality (SENASICA) reported a new case of New World Screwworm (NWS) in Ixhuatlan de Madero, Veracruz in Mexico, which is approximately 160 miles northward of the current sterile fly dispersal grid, on the eastern side of the country and 370 miles south of the U.S./Mexico border. This new northward detection comes approximately two months after northern detections were reported in Oaxaca and Veracruz, less than 700 miles away from the U.S. border, which triggered the closure of our ports to Mexican cattle, bison, and horses on May 11, 2025.

While USDA announced a risk-based phased port re-opening strategy for cattle, bison, and equine from Mexico beginning as early as July 7, 2025, this newly reported NWS case raises significant concern about the previously reported information shared by Mexican officials and severely compromises the outlined port reopening schedule of five ports from July 7-September 15. Therefore, in order to protect American livestock and our nation's food supply, Secretary Rollins has ordered the closure of livestock trade through southern ports of entry effective immediately.

"The United States has promised to be vigilant — and after detecting this new NWS case, we are pausing the planned port reopening's to further quarantine and target this deadly pest in Mexico. We must see additional progress combatting NWS in Veracruz and other nearby Mexican states in order to reopen livestock ports along the Southern border," said U.S. Secretary of Agriculture Brooke L. Rollins. "Thanks to the aggressive monitoring by USDA staff in the U.S. and in Mexico, we have been able to take quick and decisive action to respond to the spread of this deadly pest."
""".strip()

    # Random Korean text (to test non-English compression)
    korean_text = r"""
정직한 사실 위에, 공정한 시선을 더하다
Herald Korea Times

헤럴드코리아타임즈는 정치, 경제, 사회, 문화 등 한국 사회 전반의 주요 이슈를 심도 있게 다루는 종합 온라인 신문사입니다.

우리는 단순히 뉴스를 전달하는 것이 아니라, 사실(Fact)에 기반한 양측의 시각을 균형 있게 조명하며, 독자 여러분이 스스로 판단할 수 있는 '정보의 균형'을 제공합니다.

한국 언론의 오랜 문제로 지적되어 온 정치적 편향, 이념적 왜곡에서 벗어나
오직 정직함과 공정함을 원칙으로 삼는 언론을 지향합니다.
어느 한쪽의 주장만을 확대하거나 감추지 않고,
**모든 쟁점에 대해 '무엇이 쟁점인지', '누가 무엇을 주장하는지', '사실은 무엇인지'**를 명확히 전달하는 데 집중합니다.
""".strip()

    # Random piece of code
    code_text = r"""
class BasicTokenizer(Tokenizer):

    def __init__(self):
        super().__init__()

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # input text preprocessing
        text_bytes = text.encode("utf-8") # raw bytes
        ids = list(text_bytes) # list of integers in range 0..255

        # iteratively merge the most common pairs to create new tokens
        merges = {} # (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)} # int -> bytes
        for i in range(num_merges):
            # count up the number of times every consecutive pair appears
            stats = get_stats(ids)
            # find the pair with the highest count
            pair = max(stats, key=stats.get)
            # mint a new token: assign it the next available id
            idx = 256 + i
            # replace all occurrences of pair in ids with idx
            ids = merge(ids, pair, idx)
            # save the merge
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            # prints
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")
""".strip()

    math_text = r"""
\documentclass[12pt]{article}
\usepackage{amsmath,amsthm,amssymb}
\usepackage[margin=1in]{geometry}

\newtheorem{theorem}{Theorem}
\newtheorem*{remark}{Remark}

\begin{document}

\begin{center}
{\Large A Cute Identity: The Sum of Cubes is a Square}
\end{center}

\begin{theorem}
For every integer $n \ge 1$,
\[
\sum_{k=1}^{n} k^{3} \;=\; \left(\frac{n(n+1)}{2}\right)^{2}.
\]
\end{theorem}

\begin{proof}[Proof 1 (Induction)]
Let $S(n) = \sum_{k=1}^{n} k^3$. For $n=1$, $S(1)=1=(1\cdot 2/2)^2$, so the base case holds.

Assume $S(n)=\big(\tfrac{n(n+1)}{2}\big)^2$ for some $n\ge 1$.
Then
\[
S(n+1)
= S(n) + (n+1)^3
= \left(\frac{n(n+1)}{2}\right)^2 + (n+1)^3.
\]
Factor out $(n+1)^2$:
\[
S(n+1)
= (n+1)^2\left( \frac{n^2}{4} + (n+1) \right)
= (n+1)^2\left( \frac{n^2 + 4n + 4}{4} \right)
= (n+1)^2\left( \frac{(n+2)^2}{4} \right).
\]
Thus
\[
S(n+1)=\left(\frac{(n+1)(n+2)}{2}\right)^2,
\]
which matches the claimed formula with $n$ replaced by $n+1$. By induction, the identity holds for all $n\ge 1$.
\end{proof}

\begin{proof}[Proof 2 (Algebraic telescoping)]
Recall the binomial identity
\[
(k+1)^4 - k^4 = 4k^3 + 6k^2 + 4k + 1.
\]
Summing both sides from $k=0$ to $n$ telescopes:
\[
(n+1)^4 - 0^4
= \sum_{k=0}^{n}\big(4k^3 + 6k^2 + 4k + 1\big)
= 4\sum_{k=1}^{n}k^3 + 6\sum_{k=1}^{n}k^2 + 4\sum_{k=1}^{n}k + (n+1).
\]
Using the standard sums
\[
\sum_{k=1}^{n}k = \frac{n(n+1)}{2}
\quad\text{and}\quad
\sum_{k=1}^{n}k^2 = \frac{n(n+1)(2n+1)}{6},
\]
solve for $\sum_{k=1}^{n}k^3$ to get
\[
\sum_{k=1}^{n}k^3 = \left(\frac{n(n+1)}{2}\right)^2.
\]
\end{proof}

\begin{remark}
Geometrically, the identity says: ``adding up $1^3,2^3,\dots,n^3$ builds a perfect square''—namely the square of the $n$th triangular number. This is why one sometimes calls it the \emph{sum-of-cubes is a square} phenomenon.
\end{remark}

\end{document}
""".strip()

    science_text = r"""
Photosynthesis is a photochemical energy transduction process in which light-harvesting pigment–protein complexes within the thylakoid membranes of oxygenic phototrophs absorb photons and initiate charge separation at the reaction center, driving the linear electron transport chain from water to NADP⁺ via photosystem II, the cytochrome b₆f complex, and photosystem I, concomitantly generating a trans-thylakoid proton motive force utilized by chloroplastic ATP synthase. The light-dependent reactions produce ATP and NADPH, which fuel the Calvin–Benson–Bassham cycle in the stroma, wherein ribulose-1,5-bisphosphate is carboxylated by ribulose-1,5-bisphosphate carboxylase/oxygenase (RuBisCO) to form 3-phosphoglycerate, subsequently reduced and regenerated through a series of enzymatic steps, enabling net assimilation of CO₂ into triose phosphates and ultimately carbohydrates. This process is tightly regulated by photoprotective mechanisms, redox feedback, and metabolite flux, representing a central biochemical pathway coupling solar energy capture to the biosphere's primary productivity.
""".strip()

    return {
        "news": news_text,
        "korean": korean_text,
        "code": code_text,
        "math": math_text,
        "science": science_text,
    }


def print_results(tokenizer_results, vocab_sizes):
    """Print comparison results in a formatted table."""
    print("\n" + "=" * 80)
    print("TOKENIZER EVALUATION RESULTS")
    print("=" * 80)

    for tokenizer_name in tokenizer_results:
        print(
            f"\n{tokenizer_name.upper()} (vocab size: {vocab_sizes[tokenizer_name]:,})"
        )
        print("-" * 60)

        for text_name, metrics in tokenizer_results[tokenizer_name].items():
            print(
                f"  {text_name:15s}: {metrics['tokens']:6d} tokens | "
                f"{metrics['bytes']:7d} bytes | ratio: {metrics['ratio']:.3f}"
            )

    print("\n" + "=" * 80)


def main():
    args = parse_args()

    print("Loading test texts...")
    test_texts = get_test_texts()

    # The tokenizer was trained on data from earlier shards, so it has seen this data
    if not args.skip_fwe:
        print("Loading training and validation data...")
        train_docs = next(parquets_iter_batched(split="train", data_dir=args.data_dir))
        train_text = "\n".join(train_docs)
        test_texts["fwe-train"] = train_text

        val_docs = next(parquets_iter_batched(split="val", data_dir=args.data_dir))
        val_text = "\n".join(val_docs)
        if val_text:
            test_texts["fwe-val"] = val_text

    # Try out current default compared to GPT-2 and GPT-4 tokenizers
    tokenizer_results = {}
    vocab_sizes = {}

    for tokenizer_name in args.tokenizers:
        print(f"\nEvaluating {tokenizer_name} tokenizer...")

        if tokenizer_name == "gpt2":
            tokenizer = BPETokenizer.from_pretrained("gpt2")
        elif tokenizer_name == "gpt4":
            tokenizer = BPETokenizer.from_pretrained("cl100k_base")
        else:
            print(f"Loading tokenizer from {args.custom_tokenizer_model}...")
            tokenizer = BPETokenizer.from_directory(args.custom_tokenizer_model)

        vocab_sizes[tokenizer_name] = tokenizer.get_vocab_size()
        tokenizer_results[tokenizer_name] = {}

        for name, text in get_test_texts().items():
            encoded = tokenizer.encode_batch([text])[0]
            decoded = tokenizer.decode(encoded)
            assert decoded == text

            encoded_bytes = text.encode("utf-8")
            ratio = len(encoded_bytes) / len(encoded)
            tokenizer_results[tokenizer_name][name] = {
                "bytes": len(encoded_bytes),
                "tokens": len(encoded),
                "ratio": ratio,
            }

    print_results(tokenizer_results, vocab_sizes)


if __name__ == "__main__":
    main()
