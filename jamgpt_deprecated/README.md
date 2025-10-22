# JamGPT

Implemented from 0 to hero based on multiple other repositories and videos such as NanoGPT and LLM from Scratch.

```
conda create -n jamgpt python=3.13 -y
conda activate jamgpt

pip install datasets pyarrow tiktoken maturin torch pytest tokenizers requests
```

## Tokenizer

```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

maturin develop --release --manifest-path jamgpt/tokenizer/Cargo.toml

python -m scripts.train_tokenizer --max_chars=2000000000 --output_dir="./output/tokenizer"
```
