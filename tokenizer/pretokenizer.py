import regex

PATTERN = regex.compile(
    r"""'(?:s|t|re|ve|m|ll|d)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+"""
)

def pre_tokenize(text):
    return PATTERN.findall(text)


if __name__ == "__main__":
    text = "Hello, my name is Dante! I'm learning AI in 2026."
    print(pre_tokenize(text))
