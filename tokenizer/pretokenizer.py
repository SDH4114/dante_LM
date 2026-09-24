import re

def pre_tokenize(text):
    pattern = r"\s+|[^\W\d_]+|\d+|[^\w\s]"
    return re.findall(pattern, text)


if __name__ == "__main__":
    text = "Hello, my name is Dante! I have 2 cats."
    print(pre_tokenize(text))
