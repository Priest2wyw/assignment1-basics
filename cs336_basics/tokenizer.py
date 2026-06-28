import pickle
import regex as re
from collections.abc import Iterable,  Iterator

from tqdm import tqdm
import numpy as np
from numpy.lib.format import open_memmap


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
class Tokenizer():
    def __init__(self, vocab, merges, special_tokens=None):
        """
        Construct a tokenizer from a given
        vocabulary, list of merges, and (optionally) a list of special tokens.

        vocab: dict[int, bytes]  
        merges: list[tuple[bytes, bytes]]  
        special_tokens: list[str] | None = None  
        """
        self.vocab: dict[int, bytes] = vocab
        self.merges: list[tuple[bytes, bytes]] = merges
        self.special_tokens: list[str] | None = special_tokens

        self.byte_to_token_id = {v:k for k,v in self.vocab.items()}

    @classmethod
    def from_files(cls, vocab_filepath: str,
                merges_filepath: str,  
                special_tokens: list[str] | None = None
                ):
        """constructs and returns a tokenizer from a serialized vocabulary and list of merges (in the
        same format that your bpe training code output) and (optionally) a list of special tokens.
        this method should accept the following additional parameters:

            vocab_filepath: str
            merges_filepath: str  
            special_tokens: list[str] | None = None
        """
        with open(vocab_filepath, 'rb') as v_f:
            vocab = pickle.load(v_f)
        with open(merges_filepath, 'rb') as m_f:
            merges = pickle.load(m_f)
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def encode(self, text: str) -> list[int]:
        """Encode an input text into a sequence of token IDs.
        this task is find the shortest encode squenes by vocab dict?

        tras text.encode('utf-8') to id by the byte pair rank of self.merge
        train decide the sort, encode follow the sort
        """
        tokens = [] 
        if self.special_tokens:
            split_pat = "(" + "|".join(re.escape(token) 
                        for token in sorted(self.special_tokens, key=len, reverse=True)) + \
            ")"
            splited_chunk = re.split(split_pat, text)
        else:
            splited_chunk = [text]
        
        for chunk in splited_chunk:
           if self.special_tokens is not None and chunk in self.special_tokens:
               tokens.append(self.byte_to_token_id[chunk.encode('utf-8')]) 
           else:
               tokens.extend(self._encode_text(chunk))
        
        return tokens

    def _encode_text(self, string: str):

        # pre-token: for pass test_encode_special_token_double_newline_non_whitespace
        pre_tokens = []
        for m in re.finditer(PAT, string):
            word = m.group(0)
            pre_tokens.append(word)
        # 一世英名，毁于一旦啊
        # indices = list(map(int, string.encode("utf-8")))  

        token_ids = []
        for pre_token in pre_tokens:
            indices = [self.byte_to_token_id[bytes([byte])] 
                    for byte in pre_token.encode('utf-8')]

            for rank, pair in enumerate(self.merges):  
                indices = self._merge(indices, pair, rank)  
            token_ids.extend(indices)
        return token_ids

    def _merge(self, indices: list[int], pair, rank:int):
        new_indices = []
        i = 0 
        leng = len(indices)
        byte1, byte2 = pair
        token1 = self.byte_to_token_id[byte1]
        token2 = self.byte_to_token_id[byte2]

        while i < leng:
            if i < leng - 1 and indices[i] == token1 and indices[i+1] == token2:
                merged_token_id = self.byte_to_token_id[byte1+byte2]
                new_indices.append(merged_token_id)
                i += 2
            else:
                new_indices.append(indices[i])
                i += 1
        return new_indices

    # def _find_new_token_id(self, bt_idx1:int, bt_idx2:int)->int:
    #     bytes1 = self.vocab[bt_idx1]
    #     bytes2 = self.vocab[bt_idx2]
    #     merged_bytes = bytes1 + bytes2
    #     merged_bytes_token_id = self.byte_to_token_id[merged_bytes]
    #     return merged_bytes_token_id

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Given an iterable of strings (e.g., a Python file handle), 
        return a generator that lazily yields token IDs. This is
        required for memory-efficient tokenization of large files 
        that we cannot directly load into memory.
        """
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        """Decode a sequence of token IDs into text"""
        str_bytes = list(map(self.vocab.get, ids))
        strs = b''.join(str_bytes).decode('utf-8', errors="replace")
        return strs

def encode_txt_as_numpy_array(tokenizer, path_to_txt, save_path):
    with open(path_to_txt, 'r') as f:
        num_lines = sum(1 for _ in f)
    
    # 第一步：统计总token数（需要遍历一遍）
    total_tokens = 0
    with open(path_to_txt, 'r') as f:
        for line in tqdm(f, total=num_lines, desc="Counting tokens"):
            total_tokens += len(tokenizer.encode(line))

    # 第二步：创建memmap
    dtype = np.int32
    tokens_mm = open_memmap(save_path, dtype=dtype, mode='w+', shape=(total_tokens,))

    # 第三步：再次遍历写入
    pos = 0
    with open(path_to_txt, 'r') as f:
        for line in tqdm(f, total=num_lines, desc="Tokenizing"):
            ids = tokenizer.encode(line)
            n = len(ids)
            tokens_mm[pos:pos+n] = ids
            pos += n

    tokens_mm.flush()

if __name__ == "__main__":
    import os, pathlib
    TOKENIZER_DIR = pathlib.Path(__file__).resolve().parent.parent / "tokenizer"
    VOCAB_PATH = os.path.join(TOKENIZER_DIR, "tinystories_bpe_vocab.pkl")
    MERGES_PATH = os.path.join(TOKENIZER_DIR, "tinystories_bpe_merges.pkl")

    tokenizer =  Tokenizer.from_files(
        vocab_filepath=VOCAB_PATH,
        merges_filepath=MERGES_PATH,
        special_tokens=["<|endoftext|>"]
        )

    test_texts = [
        "我爱北京天安门<|endoftext|>",
        "this is a test",
    ]
    
    encoded_texts = [tokenizer.encode(tx) for tx in test_texts]
    decoded_txts = [tokenizer.decode(et) for et in encoded_texts]

    for t, e, d in zip(test_texts, encoded_texts, decoded_txts):
        print(f"原始文本为{t},\n encode_token is {e},\n decoded_text is {d}")
    assert decoded_txts == test_texts
    
    process_data = True
    if process_data:
        DATA_DIR = pathlib.Path(__file__).resolve().parent.parent
        TRAIN_TXT_DATA_PATH=os.path.join(DATA_DIR, "TinyStoriesV2-GPT4-train.txt")
        VAL_TXT_DATA_PATH=  os.path.join(DATA_DIR, "TinyStoriesV2-GPT4-valid.txt")
        TRAIN_DATA_PATH=    os.path.join(DATA_DIR, "data/TinyStoriesV2-GPT4-train.npy")
        VAL_DATA_PATH=      os.path.join(DATA_DIR, "data/TinyStoriesV2-GPT4-valid.npy")
        # encode_txt_as_numpy_array(tokenizer, TRAIN_TXT_DATA_PATH, TRAIN_DATA_PATH)
        encode_txt_as_numpy_array(tokenizer, VAL_TXT_DATA_PATH, VAL_DATA_PATH)