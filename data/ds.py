from datasets import load_dataset

def load_data():
    ds = load_dataset("HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
        cache_dir="datasets/cache")
    return ds

def get_text(ds):
    for sample in ds:
        yield sample["text"]
# yield - вернул значение и поставил функцию на паузу
