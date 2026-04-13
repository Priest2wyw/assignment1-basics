import pickle
from pathlib import Path
from tests.adapters import run_train_bpe
from tests.common import FIXTURES_PATH

# 1. 加载快照
snapshot_path = Path(__file__).parent / "_snapshots" / "test_train_bpe_special_tokens.pkl"
with open(snapshot_path, "rb") as f:
    expected_data = pickle.load(f)

# 2. 运行训练生成实际词汇
input_path = FIXTURES_PATH / "tinystories_sample_5M.txt"
vocab, merges = run_train_bpe(
    input_path=input_path,
    vocab_size=1000,
    special_tokens=["<|endoftext|>"],
)

actual_data = {
    "vocab_keys": set(vocab.keys()),
    "vocab_values": set(vocab.values()),
    "merges": merges,
}

# 3. 对比分析
print("=== 词汇键对比 ===")
expected_keys = expected_data["vocab_keys"]
actual_keys = actual_data["vocab_keys"]
print(f"预期词汇键数: {len(expected_keys)}")
print(f"实际词汇键数: {len(actual_keys)}")
print(f"缺失键: {expected_keys - actual_keys}")
print(f"多余键: {actual_keys - expected_keys}")

print("\n=== 词汇值对比 ===")
expected_values = expected_data["vocab_values"]
actual_values = actual_data["vocab_values"]
print(f"预期词汇值数: {len(expected_values)}")
print(f"实际词汇值数: {len(actual_values)}")

# 关键diff
missing_values = expected_values - actual_values
extra_values = actual_values - expected_values

print(f"\n缺失的值 ({len(missing_values)}):")
for v in sorted(missing_values)[:10]:  # 只显示前10个
    print(f"  {v}")
if len(missing_values) > 10:
    print(f"  ... 还有 {len(missing_values) - 10} 个")

print(f"\n多余的值 ({len(extra_values)}):")
for v in sorted(extra_values)[:10]:  # 只显示前10个
    print(f"  {v}")
if len(extra_values) > 10:
    print(f"  ... 还有 {len(extra_values) - 10} 个")

# 4. 检查初始化词汇
print("\n=== 检查初始化词汇 ===")
all_actual_values = set(vocab.values())
single_bytes = {bytes([i]) for i in range(256)}
special_tokens_bytes = {b"<|endoftext|>"}
expected_initial = single_bytes | special_tokens_bytes

print(f"初始化词汇是否完整: {expected_initial.issubset(all_actual_values)}")

# 5. 对比合并序列
print("\n=== 合并序列对比 ===")
expected_merges = expected_data["merges"]
print(f"预期合并数: {len(expected_merges)}")
print(f"实际合并数: {len(actual_merges := merges)}")

if expected_merges != actual_merges:
    for i, (exp, act) in enumerate(zip(expected_merges, actual_merges)):
        if exp != act:
            print(f"第 {i} 个合并不同:")
            print(f"  预期: {exp}")
            print(f"  实际: {act}")
            if i >= 5:  # 只显示前5个差异
                print("  ...")
                break
