import pickle

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

    def from_files(self, 
                vocab_filepath: str,
                merges_filepath: str,  
                special_tokens: list[str] | None = None
                ):
        """constructs and returns a Tokenizer from a serialized vocabulary and list of merges (in the
        same format that your BPE training code output) and (optionally) a list of special tokens.
        This method should accept the following additional parameters:
        """
        return NotImplemented
    def encode(self, text: str) -> list[int]:
        """Encode an input text into a sequence of token IDs."""
        return NotImplemented

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Given an iterable of strings (e.g., a Python file handle), 
        return a generator that lazily yields token IDs. This is
        required for memory-efficient tokenization of large files 
        that we cannot directly load into memory.
        """
        return NotImplemented

    def decode(self, ids: list[int]) -> str:
        """Decode a sequence of token IDs into text"""
        return NotImplemented