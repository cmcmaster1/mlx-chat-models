from huggingface_hub import HfApi
import json
from transformers.utils.hub import cached_file
from datasets import Dataset, load_dataset, concatenate_datasets
import os

def has_chat_template(model_id):
    try:
        tokenizer_config = cached_file(model_id, "tokenizer_config.json")
        with open(tokenizer_config, "r") as f:
            return 'chat_template' in json.load(f)
    except Exception:
        return False
    
def mlx_chat_models(current_models):
    current_models = current_models["model_id"]
    api = HfApi()
    models = api.list_models(
        author="mlx-community",
    )
    models = list(models)
    models = [model for model in models if model not in current_models]
    chat_model = [has_chat_template(model.id) for model in models]
    dataset = Dataset.from_dict({"model_id": [model.id for model in models], "chat_model": chat_model})
    return dataset

def main():
    current_models = load_dataset("cmcmaster/mlx-chat-models")["train"]
    new_models = mlx_chat_models(current_models)
    updated_models = concatenate_datasets([current_models, new_models])

    updated_models.push_to_hub("cmcmaster/mlx-chat-models", token=os.environ["HF_TOKEN"])

if __name__ == "__main__":
    main()
