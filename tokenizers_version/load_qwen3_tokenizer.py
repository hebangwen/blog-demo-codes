import json

from huggingface_hub import hf_hub_download
import tokenizers
from tokenizers import Tokenizer


print(f"{tokenizers.__version__ = }")


def load_tokenizer_and_transformers_version(model_id):
    tokenizer_filepath = hf_hub_download(repo_id=model_id, filename="tokenizer.json")
    try:
        tokenizer = Tokenizer.from_file(tokenizer_filepath)
        print(tokenizer)
    except Exception as e:
        print(f"Error loading tokenizer for model {model_id}: {e}")

    config_filepath = hf_hub_download(repo_id=model_id, filename="config.json")
    config = json.load(open(config_filepath, "r", encoding="utf-8"))
    record_version = config.get("transformers_version", "unknown")
    print(f"Model ID: {model_id}, Transformers Version: {record_version}")

    # if model_id == QWEN3_MODEL_ID and tokenizers.__version__ == "0.21.1":
    #     from transformers import AutoTokenizer
    #     tokenizer = AutoTokenizer.from_pretrained(model_id)
    #     tokenizer.save_pretrained("qwen3_tokenizer")


QWEN3_MODEL_ID = "Qwen/Qwen3-8B"
load_tokenizer_and_transformers_version(QWEN3_MODEL_ID)

QWEN2_5_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
load_tokenizer_and_transformers_version(QWEN2_5_MODEL_ID)
