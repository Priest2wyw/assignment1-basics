import os
import regex as re
from time import time 
from copy import deepcopy
from typing import BinaryIO
from collections import Counter
from collections import defaultdict

from tqdm import tqdm

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

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
    # profile process 1: replace counter class with dafaultdict for count number
    # this can cut down 45% of all procss 
    total_counter = defaultdict(int)
    with open(input_path, 'rb') as f:
        num_processes = 16
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
    
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            splited_chunks = re.split("<|endoftext|>", chunk)
            for sck in splited_chunks:
                splited_words = split_chunk(sck)
                for word in splited_words:
                    total_counter[word] += 1
        
        freq_table = trans_table_from_words_to_bytes(dict(total_counter))
    return  freq_table
        
# def get_pair_freqs(freq_table:dict[tuple[int,...], int]
                #    ) -> dict[tuple[int, int], int]:
    # """get count of pair like (index1, index2)"""
    # counter = defaultdict(int)
    # for byte_indexs, count in freq_table.items():
        # for byte_idx1, byte_idx2 in zip(byte_indexs, byte_indexs[1:]):
            # counter[(byte_idx1, byte_idx2)] += count
            # pair_cache[(byte_idx1, byte_idx2)].append(byte_indexs)
    # return counter
        
 
def init_pair_idx_and_count_cache(freq_table:dict[tuple[int,...], int]
                   ) -> dict[tuple[int, int], 
                             list[tuple[int,...,int],...]]:
    """get count of pair like (index1, index2)"""
    pair_cache = defaultdict(list)
    counter_cache = defaultdict(int)
    for byte_indexs, count in freq_table.items():
        for byte_idx1, byte_idx2 in zip(byte_indexs, byte_indexs[1:]):
            pair_cache[(byte_idx1, byte_idx2)].append(byte_indexs)
            counter_cache[(byte_idx1, byte_idx2)] += count
    return pair_cache, counter_cache

def merge_and_update_cache(freq_table: dict[tuple[int, ..., int], int], 
          pair: tuple[int, int], 
          new_index: int,
          pair_idx_cache, 
          pair_count_cache):
    new_freq_table = deepcopy(freq_table)
    new_pair_idx_cache = deepcopy(pair_idx_cache)
    new_pair_count_cache = deepcopy(pair_count_cache)

    byte_1, byte_2 = pair

    # merge freq table 
    update_token_ids = pair_idx_cache[pair]
    updated_idx_caches = []
    for token_ids in update_token_ids:
        count = new_freq_table.pop(token_ids) # (1,2,3):5
        new_token_ids = []
        i = 0
        token_len = len(token_ids)
            
        while i < token_len:
            if i < token_len-1 and token_ids[i] == byte_1 and token_ids[i+1] == byte_2:
                new_token_ids.append(new_index)
                i += 2
            else:
                new_token_ids.append(token_ids[i]) 
                i += 1
        new_token_ids = tuple(new_token_ids)
        new_freq_table[new_token_ids] = count
        updated_idx_caches.append((token_ids, new_token_ids, count))

    # not finish
    # update idx_cache and count_cache
    for old_token_ids, new_token_ids, count in updated_idx_caches:
        # ✅ 第一步：从旧token的所有对中【减去计数】并【移除索引】
        old_len = len(old_token_ids)
        for j in range(old_len - 1):
            p = (old_token_ids[j], old_token_ids[j+1])
            # 减去这个token对该对的贡献
            if p in new_pair_count_cache:
                new_pair_count_cache[p] -= count
                if new_pair_count_cache[p] <= 0:
                    del new_pair_count_cache[p]
            # 从索引中移除旧token
            if p in new_pair_idx_cache and old_token_ids in new_pair_idx_cache[p]:
                new_pair_idx_cache[p].remove(old_token_ids)
                if not new_pair_idx_cache[p]:  # 如果这个pair列表为空
                    del new_pair_idx_cache[p]
        
        # ✅ 第二步：给新token的所有对【加上计数】并【添加索引】
        new_len = len(new_token_ids)
        for j in range(new_len - 1):
            p = (new_token_ids[j], new_token_ids[j+1])
            new_pair_count_cache[p] = new_pair_count_cache.get(p, 0) + count
            new_pair_idx_cache[p].append(new_token_ids)
    
    # 删除已处理完毕的pair
    if pair in new_pair_count_cache:
        del new_pair_count_cache[pair]
    if pair in new_pair_idx_cache:
        del new_pair_idx_cache[pair]

    return new_freq_table, new_pair_idx_cache, new_pair_count_cache
            
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
    print(">>> 1. start pre-token")
    start_time = time()
    freq_table = pre_tokenize(input_path) 
    end_time = time()
    print(f'len of freq_table is {len(freq_table)}')
    print(f">>> 1. end pre-token, time cost is {end_time - start_time}s")

    # 4. merge: caching procedure
    item_step = max(vocab_size - len(vocab), 0)
    pair_idx_cache, pair_count_cache = init_pair_idx_and_count_cache(freq_table)
    for i in tqdm(range(item_step)):
    # for i in tqdm(range(min(100, item_step))):
        # get 最高频数的pair
        # most time custer:50% of tottime, 30% of cumtime
        # pair_freqs = get_pair_freqs(freq_table, pair_cache)

        # # bad inflent
        # pair = max(pair_freqs, key=pair_freqs.get) 
         
        # must get the same result using
        # `preferring the lexicographically greater pair`
        max_count = max(pair_count_cache.values()) 
        max_count_indx_pairs = [pair for pair, count in pair_count_cache.items() if count==max_count]
        byte_index_pair = max(max_count_indx_pairs, key=lambda p: (vocab[p[0]], vocab[p[1]] ))
        byte_idx1, byte_idx2 = byte_index_pair

        # merge
        new_index = len(vocab) 
        merges.append((vocab[byte_idx1], vocab[byte_idx2])) 

        ## TODO: if the order of special_token is first, how to finish new one
        vocab[new_index] = vocab[byte_idx1] + vocab[byte_idx2]
        freq_table, pair_idx_cache, pair_count_cache = merge_and_update_cache(
            freq_table, byte_index_pair, 
            new_index, pair_idx_cache, pair_count_cache
            )

    return vocab, merges

if __name__ == "__main__":
    # input_path = "/home/youwei/github/cs336/assignment1-basics/TinyStoriesV2-GPT4-valid.txt"
    input_path = "/home/youwei/github/cs336/assignment1-basics/TinyStoriesV2-GPT4-train.txt"
    # input_path = "/home/youwei/github/cs336/assignment1-basics/test_dataset_800k.txt"
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)
