import regex as re
from collections.abc import Iterable,  Iterator

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

    def from_files(self, 
                vocab_filepath: str,
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
        return NotImplemented

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