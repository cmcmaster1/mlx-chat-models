import os
import json
from typing import Optional

from datasets import Dataset, load_dataset, concatenate_datasets
from huggingface_hub import HfApi
from transformers.utils.hub import cached_file

def has_chat_template(model_id: str) -> bool:
    try:
        tokenizer_config = cached_file(model_id, "tokenizer_config.json")
        with open(tokenizer_config, "r") as f:
            return 'chat_template' in json.load(f)
    except Exception as e:
        print(f"Error checking chat template for {model_id}: {e}")
        return False

def get_mlx_chat_models(current_models: Optional[Dataset] = None) -> Dataset:
    api = HfApi()
    models = [model.id for model in api.list_models(author="mlx-community")]
    
    if current_models is not None:
        current_model_ids = set(current_models["model_id"])
        new_models = [model for model in models if model not in current_model_ids]
        print(f"Found {len(new_models)} new models")
    else:
        new_models = models
        print(f"Found {len(new_models)} models")

    chat_model_status = [has_chat_template(model) for model in new_models]
    return Dataset.from_dict({"model_id": new_models, "chat_model": chat_model_status})

def main():
    try:
        current_models = load_dataset("cmcmaster/mlx-chat-models")["train"]
        print("Loaded existing dataset")
    except Exception as e:
        print(f"Error loading existing dataset: {e}")
        current_models = None

    new_models = get_mlx_chat_models(current_models)
    
    if current_models is not None:
        updated_models = concatenate_datasets([current_models, new_models])
    else:
        updated_models = new_models

    print(f"Total models: {len(updated_models)}")

    try:
        updated_models.push_to_hub("cmcmaster/mlx-chat-models", token=os.environ["HF_TOKEN"])
        print("Successfully updated dataset on Hugging Face Hub")
    except Exception as e:
        print(f"Error pushing dataset to Hugging Face Hub: {e}")
        raise

if __name__ == "__main__":
    main()
