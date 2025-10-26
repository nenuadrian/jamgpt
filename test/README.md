

```bash
python train_tokenizer.py --dataset_path /Volumes/StorageT4/data/fineweb-edu-parquet-shards/sample-100BT --vocab_size 50257

python test_tokenizer.py --custom_tokenizer_path ./tokenizer_output/tokenizer.json

# Train a small GPT model
python train_gpt.py \
    --dataset_path /Volumes/StorageT4/data/fineweb-edu-parquet-shards/sample-100BT \
    --tokenizer_path ./tokenizer_output/tokenizer.json \
    --output_dir ./gpt_output \
    --max_files 5 \
    --batch_size 16 \
    --max_iters 5000

# Train with more capacity
python train_gpt.py \
    --n_layer 12 \
    --n_head 12 \
    --n_embd 768 \
    --block_size 512 \
    --batch_size 8 \
    --dataset_path /Volumes/StorageT4/data/fineweb-edu-parquet-shards/sample-100BT \
    --tokenizer_path ./tokenizer_output/tokenizer.json \
    --output_dir ./gpt_output 


# 1. Create a chat dataset from HuggingFace
python create_chat_dataset.py \
    --dataset_name tatsu-lab/alpaca \
    --output_path ./chat_data.jsonl \
    --max_samples 10000

# 2. Finetune pretrained model for chat
python train_chat.py \
    --pretrained_model_path ./gpt_output/best_model.pt \
    --tokenizer_path ./tokenizer_output/tokenizer.json \
    --dataset_path ./chat_data.jsonl \
    --output_dir ./chat_gpt_output \
    --batch_size 8 \
    --learning_rate 1e-5 \
    --max_iters 2000

```