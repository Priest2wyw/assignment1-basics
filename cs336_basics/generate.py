'''
uv run cs336_basics/generate.py  \
    --tokenizer_path tokenizer/ \
    --model_path ./models/lr_tune/ \ 
    --input "good morning, 你好 " \
    --max_generate_token 1024
'''
import os
import argparse
import torch
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.model import BasicsTransformerLM

def parse_arg():
    parser = argparse.ArgumentParser(description="输入")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="tokenizer文件位置")
    parser.add_argument("--model_path", type=str, required=True, help="模型文件位置")
    parser.add_argument("--max_generate_token", type=int, default=256, help="生成token数量")
    parser.add_argument("--input", type=str, required=True, help="输入需要生成的文本")

    args = parser.parse_args()

    return args

def main():
    args = parse_arg()
    
    # tokenizer init 
    vocab_path  = os.path.join(args.tokenizer_path, "tinystories_bpe_vocab.pkl")
    merges_path = os.path.join(args.tokenizer_path, "tinystories_bpe_merges.pkl")

    tokenizer =  Tokenizer.from_files(
        vocab_filepath=vocab_path,
        merges_filepath=merges_path,
        special_tokens=["<|endoftext|>"]
    )
    
    # model init 
    if os.path.exists(args.model_path):
        model = BasicsTransformerLM.from_pretrained(args.model_path)
    else:
        raise f"model_path not exist: { args.model_path}"
    # generate
    input_tokens = tokenizer.encode(args.input)
    input_tokens = torch.tensor(input_tokens)
    print("开始生成:>")
    output_tokens = model.generate(input_tokens, max_new_tokens=args.max_generate_token)

    out_str = tokenizer.decode(output_tokens[0].tolist())
    print(out_str)
   
if __name__ == '__main__':
    main()