#!/bin/bash
# Profile Tokenization speed using Gigatoken CLI

if [ -z "$1" ]; then
  echo "Usage: $0 <path_to_dataset.csv_or_txt>"
  exit 1
fi

DATASET=$1

echo "=============================================="
echo "Profiling HuggingFace GPT-2 Tokenizer vs Gigatoken"
echo "=============================================="
uvx --with tokenizers gigatoken bench 'openai-community/gpt2' $DATASET --validate

echo ""
echo "=============================================="
echo "Profiling OpenAI cl100k_base (Tiktoken) vs Gigatoken"
echo "=============================================="
uvx --with tokenizers gigatoken bench 'cl100k_base' $DATASET --validate
