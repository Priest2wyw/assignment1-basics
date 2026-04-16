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
        
def init_pair_idx_and_count_cache(freq_table: dict[tuple[int, ...], int]):
    """
    初始化两个缓存：
    - pair_count_cache[pair] = 该 pair 在整个语料中的加权总出现次数
    - pair_idx_cache[pair]   = Counter，键是包含该 pair 的 token_ids，
                               值是该 pair 在该 token_ids 内出现的次数
    """
    pair_count_cache: dict[tuple[int, int], int] = defaultdict(int)
    pair_idx_cache: dict[tuple[int, int], Counter] = defaultdict(Counter)

    for token_ids, freq in freq_table.items():
        for j in range(len(token_ids) - 1):
            p = (token_ids[j], token_ids[j + 1])
            pair_count_cache[p] += freq
            pair_idx_cache[p][token_ids] += 1

    return dict(pair_idx_cache), dict(pair_count_cache)


def _remove_pair_occurrence(pair_idx_cache, pair_count_cache,
                            p, token_ids, occ_in_seq, freq):
    """从缓存里扣掉 token_ids 对 pair p 的贡献。"""
    pair_count_cache[p] -= freq * occ_in_seq
    if pair_count_cache[p] <= 0:
        del pair_count_cache[p]

    bucket = pair_idx_cache[p]
    bucket[token_ids] -= occ_in_seq
    if bucket[token_ids] <= 0:
        del bucket[token_ids]
    if not bucket:
        del pair_idx_cache[p]


def _add_pair_occurrence(pair_idx_cache, pair_count_cache,
                         p, token_ids, occ_in_seq, freq):
    """给 token_ids 对 pair p 的贡献加进缓存。"""
    pair_count_cache[p] = pair_count_cache.get(p, 0) + freq * occ_in_seq
    if p not in pair_idx_cache:
        pair_idx_cache[p] = Counter()
    pair_idx_cache[p][token_ids] += occ_in_seq


def merge_and_update_cache(freq_table: dict[tuple[int, ...], int],
                           pair: tuple[int, int],
                           new_index: int,
                           pair_idx_cache: dict,
                           pair_count_cache: dict):
    """
    合并一个 pair，就地更新 freq_table 和两个缓存。
    """
    byte_1, byte_2 = pair

    affected = list(pair_idx_cache[pair].items())

    for old_token_ids, _old_occ in affected:
        freq = freq_table.pop(old_token_ids)

        old_pairs = Counter()
        for j in range(len(old_token_ids) - 1):
            old_pairs[(old_token_ids[j], old_token_ids[j + 1])] += 1

        new_list = []
        i, n = 0, len(old_token_ids)
        while i < n:
            if i < n - 1 and old_token_ids[i] == byte_1 and old_token_ids[i + 1] == byte_2:
                new_list.append(new_index)
                i += 2
            else:
                new_list.append(old_token_ids[i])
                i += 1
        new_token_ids = tuple(new_list)

        new_pairs = Counter()
        for j in range(len(new_token_ids) - 1):
            new_pairs[(new_token_ids[j], new_token_ids[j + 1])] += 1

        freq_table[new_token_ids] = freq_table.get(new_token_ids, 0) + freq

        for p, occ in old_pairs.items():
            _remove_pair_occurrence(pair_idx_cache, pair_count_cache,
                                    p, old_token_ids, occ, freq)

        for p, occ in new_pairs.items():
            _add_pair_occurrence(pair_idx_cache, pair_count_cache,
                                 p, new_token_ids, occ, freq)

    assert pair not in pair_count_cache, f"{pair} 残留在 count cache"
    assert pair not in pair_idx_cache, f"{pair} 残留在 idx cache"

    return freq_table, pair_idx_cache, pair_count_cache


def train_bpe(input_path: str,
              vocab_size: int,
              special_tokens: list[str]
              ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab: dict[int, bytes] = {}
    merges: list[tuple[bytes, bytes]] = []

    init_tokens = [bytes([i]) for i in range(256)] + [
        st.encode('utf-8') for st in special_tokens]
    vocab = {idx: token for idx, token in enumerate(init_tokens)}

    print(">>> 1. start pre-token")
    t0 = time()
    freq_table = pre_tokenize(input_path)  # 你原来的函数
    print(f"len of freq_table is {len(freq_table)}")
    print(f">>> 1. end pre-token, time cost is {time() - t0:.2f}s")

    pair_idx_cache, pair_count_cache = init_pair_idx_and_count_cache(freq_table)

    n_merges = max(vocab_size - len(vocab), 0)
    for _ in tqdm(range(n_merges)):
        if not pair_count_cache:
            break

        max_count = max(pair_count_cache.values())
        candidates = [p for p, c in pair_count_cache.items() if c == max_count]
        byte_index_pair = max(candidates, key=lambda p: (vocab[p[0]], vocab[p[1]]))

        new_index = len(vocab)
        merges.append((vocab[byte_index_pair[0]], vocab[byte_index_pair[1]]))
        vocab[new_index] = vocab[byte_index_pair[0]] + vocab[byte_index_pair[1]]

        merge_and_update_cache(
            freq_table, byte_index_pair, new_index,
            pair_idx_cache, pair_count_cache
        )

    return vocab, merges

if __name__ == "__main__":
    # input_path = "/home/youwei/github/cs336/assignment1-basics/TinyStoriesV2-GPT4-valid.txt"
    input_path = "/home/youwei/github/cs336/assignment1-basics/TinyStoriesV2-GPT4-train.txt"
    # input_path = "/home/youwei/github/cs336/assignment1-basics/test_dataset_800k.txt"
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)
