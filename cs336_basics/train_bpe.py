import regex as re
from collections import Counter
from collections import defaultdict
from pretokenization_example import find_chunk_boundaries

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def split_chunk(chunk: str) -> list[str]:
    return re.findall(PAT, chunk)

def trans_table_from_words_to_bytes(table_freq: dict[str, int]) -> dict[tuple[bytes], int]:
    """transformer {'the': 1} -> {('t', 'h', 'e'): 1}"""
    new_freq_table = {}
    for  freq, word in enumerate(table_freq):
        new_freq_table[tuple(word)]=freq
    return new_freq_table

def pre_tokenize(input_path:str)->dict[tuple[str, ..., str], int]:
    total_counter = Counter()
    with open(input_path, 'rb') as f:
        num_processes = 8
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
    
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunk = chunk.replace("<|endoftext|>", "")
            splited_words = split_chunk(chunk)
            
            chunk_counter = Counter(splited_words)
            
            total_counter += chunk_counter 
        
        freq_table = trans_table_from_words_to_tuple(dict(total_counter))
    return  freq_table
        
def get_pair_freqs(freq_table:dict[tuple[str,...,str], int]) -> dict[tuple[str, str], int]:
    """get count of pair like (char1, char2)"""
    counter = defaultdict(int)
    for word_tupe, count in freq_table.items():
        for char1, char2 in zip(word_tupe, word_tupe[1:]):
            counter[(char1, char2)] += count
    return counter
        
    
def merge(freq_table, pair):
    """replace (char1, char2)new pair """
def  train_bpe(input_path: str,
               vocab_size: int,
               special_tokens: list[str]
               ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab: dict[int, bytes] = {}
    merges: list[tuple[bytes, bytes]] = []
    
    # 1. init vocab
    init_tokens =  [bytes([i]) for i in  range(256)] + [
        special_token.encode('utf-8') for special_token in special_tokens] 
    vocab = {idx:token for idx, token in enumerate(init_tokens)}

    # 2. parallelizing pre-tokenization
    freq_table = pre_tokenize(input_path)
    print(f'len of freq_table is {len(freq_table)}')

    # 4. merge: caching procedure
    while len(vocab) < vocab_size:
        # get 最高频数的pair
        pair_freqs = get_pair_freqs(freq_table)
        pair = max(pair_freqs, key=pair_freqs.get) 
        char1, char2 = pair
        
        
        # merge
        new_index = len(vocab) 
        merges.append(pair) 

        ## TODO: if the order of special_token is first, how to finish new one
        vocab[new_index] = vocab[byte_idx1] + vocab[byte_idx2]
        freq_table = merge(freq_table, pair, new_index)

    return vocab, merges

if __name__ == "__main__":
    input_path = "/home/youwei/github/cs336/assignment1-basics/TinyStoriesV2-GPT4-valid.txt"
    vocab_size = 1000
    special_tokens = ["<|endoftext|>"]
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)