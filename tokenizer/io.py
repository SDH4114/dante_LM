import json

def save_tokenizer(path, merges, special_tokens):
    data = {"merges": [], "special_tokens": special_tokens}

    for pair, token_id in merges.items():
        data["merges"].append({"pair": list(pair),"id": token_id})

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def load_tokenizer(path):
    with open(path,"r", encoding="utf-8") as file:
        data = json.load(file)

    merges = {}
    for item in data["merges"]:
        pair = tuple(item["pair"])
        token_id = item["id"]

        merges[pair] = token_id

    special_tokens = data["special_tokens"]

    return merges, special_tokens
