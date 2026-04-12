encoding_types = ['utf-8', 'utf-16', 'utf-32']
def test(text, encoding_type):
    encoded = text.encode(encoding_type)
    print(encoded)
    print(type(encoded))
    print(list(encoded))
    print(f"len of text is {len(text)}, len of encoding is {len(encoded)}")
    print(encoded.decode(encoding_type))
    return len(encoded)

def encode_string_len(text, encoding_type):
    encoded = text.encode(encoding_type)
    return len(encoded)

def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])

if __name__ == "__main__":
    # test("hello! こんにちは!你好 👋" ,"utf-8")

    # qestion a
    encoding_types = ['utf-8', 'utf-16', 'utf-32']
    test_strings = ["hello", "你好", "😄🐵☀️", "hello! こんにちは!你好 👋"]
    print(f"{'String':<30} {'Chars':>6} | {'utf-8':>8} {'utf-16':>8} {'utf-32':>8}")
    print('-' * 70)

    for test_string in test_strings:
        results = {}
        for encoding_type in encoding_types:
            results = {et: encode_string_len(test_string, et) for et in encoding_types}
            ratio = results['utf-8'] / results['utf-16'] if results['utf-16'] else 0
        
        print(f"{test_string:<50} {len(test_string):>6} | "
            f"{results['utf-8']:>8} {results['utf-16']:>8} {results['utf-32']:>8} | "
            f"{ratio:>19.2f}")

    answer_a = """utf-8 has less len of encoding vs utf-16 and utf-32, because utf-8 
    uses 1 byte for ASCII characters, while utf-16 and utf-32 use 2 and 4 bytes respectively for all characters
    """
    print(answer_a) 

    # question b
    try:
        decoded = decode_utf8_bytes_to_str_wrong("牛".encode('utf-8'))
        print(f"解码成功: {decoded}")
    except UnicodeDecodeError as e:
        print(f"解码失败: {e}")
    anwser_b = """
>>> ANSWER_B = decode_utf8_bytes_to_str_wrong("牛".encode('utf-8'))
The text `"牛".encode('utf-8')` input decode_utf8_bytes_to_str_wrong will return error
`UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe4 in position 0: unexpected end of data`,
because "牛".encode('utf-8') has 3 bytes, and the function decode_utf8_bytes_to_str_wrong tries to decode each byte separately, which is not valid UTF-8 encoding. 
Each character in UTF-8 can be represented by 1 to 4 bytes, and decoding them separately will lead to errors when the bytes do not form a valid character on their own.
"""
    print(anwser_b)

    # question c
    # constrcut an illegaal UTF-8 sequence：10xxxxxx (2 bytes, but start with 1110)
    illegal_bytes = bytes([0b11100000, 0b10111111])  # [0xE0, 0xBF]

    print(f"illegal sequence: {list(illegal_bytes)}")
    try:
        decoded = illegal_bytes.decode('utf-8')
        print(f"decode success: {decoded}")
    except UnicodeDecodeError as e:
        print(f"decode failed: {e}")
    
    answer_c = """ >>> ANSWER_c: 11100000 10111111
utf-8 use first 1-4 code units to determine the length of the character, 
if the first byte starts with 1110, 
it indicates that the character should be represented by 3 bytes,
but we just give 2."""
    print(answer_c)