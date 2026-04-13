import regex as re
from collections import Counter
from collections import defaultdict
from pretokenization_example import find_chunk_boundaries

from tqdm import tqdm

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def split_chunk(chunk: str) -> list[str]:
    return re.findall(PAT, chunk)

def trans_table_from_words_to_bytes(table_freq: dict[str, int]) -> dict[tuple[int], int]:
    """transforer word count to byte int count
    Args:
        table_freq: {'the': 5, 'hello': 3}
    
    Returns:
        {(116, 104, 101): 5, (104, 101, 108, 108, 111): 3}
    """
    
    return {
        tuple(word.encode('utf-8')): count
        for word, count in table_freq.items()
    }

def pre_tokenize(input_path:str)->dict[tuple[int, ..., int], int]:
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
        
        freq_table = trans_table_from_words_to_bytes(dict(total_counter))
    return  freq_table
        
def get_pair_freqs(freq_table:dict[tuple[int,...,int], int]) -> dict[tuple[int, int], int]:
    """get count of pair like (index1, index2)"""
    counter = defaultdict(int)
    for byte_indexs, count in freq_table.items():
        for byte_idx1, byte_idx2 in zip(byte_indexs, byte_indexs[1:]):
            counter[(byte_idx1, byte_idx2)] += count
    return counter
        
    
def merge(freq_table: dict[tuple[int, ..., int], int], 
          pair: tuple[int, int], 
          new_index: int):
    new_freq_table = {}
    byte_1, byte_2 = pair
    
    sub_squene_str = str(pair)[1:-1] # remove () 
    for token_ids, count in freq_table.items():
        # cache: if subsquene not in token_ids, not update 
        matched_str = str(token_ids) 
        if sub_squene_str not in matched_str:
            new_freq_table[token_ids] = count 
        else:
            i = 0
            new_token_ids = []
            while i < len(token_ids):
                if i < len(token_ids)-1 and token_ids[i] == byte_1 and token_ids[i+1] == byte_2:
                    new_token_ids.append(new_index)
                    i += 2
                else:
                    new_token_ids.append(token_ids[i]) 
                    i += 1
            new_freq_table[tuple(new_token_ids)] = count
    return new_freq_table
            
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
    item_step = max(vocab_size - len(vocab), 0)
    for i in tqdm(range(item_step)):
    # for i in tqdm(range(min(100, item_step))):
        # get 最高频数的pair
        pair_freqs = get_pair_freqs(freq_table)

        # # bad inflent
        # pair = max(pair_freqs, key=pair_freqs.get) 
        
        # must get the same result using
        # `preferring the lexicographically greater pair`
        max_count = max(pair_freqs.values()) 
        max_count_indx_pairs = [pair for pair, count in pair_freqs.items() if count==max_count]
        byte_index_pair = max(max_count_indx_pairs, key=lambda p: (vocab[p[0]], vocab[p[1]] ))
        byte_idx1, byte_idx2 = byte_index_pair

        # # debug block
        # pair_count = pair_freqs[pair]
        # if i>=90:
        #     print(f"Round {i}: chose {(vocab[byte_idx1], vocab[byte_idx2])} (count={pair_count})(max_count_pairs is {max_count_pairs})")
        
        # merge
        new_index = len(vocab) 
        merges.append((vocab[byte_idx1], vocab[byte_idx2])) 

        ## TODO: if the order of special_token is first, how to finish new one
        vocab[new_index] = vocab[byte_idx1] + vocab[byte_idx2]
        freq_table = merge(freq_table, byte_index_pair, new_index)

    return vocab, merges

if __name__ == "__main__":
    input_path = "/home/youwei/github/cs336/assignment1-basics/TinyStoriesV2-GPT4-valid.txt"
    vocab_size = 10_000
    special_tokens = ["<|endoftext|>"]
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)