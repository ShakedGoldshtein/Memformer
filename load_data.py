"""
Load WikiText-2 from Hugging Face (no manual download; data is fetched on first run).
"""
from datasets import load_dataset

def get_wikitext2(split="train"):
    """Load WikiText-2. split: 'train' | 'validation' | 'test'"""
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    return ds

if __name__ == "__main__":
    train = get_wikitext2("train")
    print(f"Train: {len(train)} lines")
    print("Sample:", train["text"][:3])
