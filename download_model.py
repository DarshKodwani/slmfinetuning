from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Define the list of models and their save paths
models_to_download = [
    {"name": "huggingface/smollm-135m-instruct", "model_path": "smollm-135m.pth", "tokenizer_path": "smollm-135m-tokenizer"},
    {"name": "google/t5-small", "model_path": "t5-small.pth", "tokenizer_path": "t5-small-tokenizer"},
    {"name": "huggingface/tiny-llama", "model_path": "tiny-llama.pth", "tokenizer_path": "tiny-llama-tokenizer"},
    {"name": "huggingface/smollm2-360m-instruct", "model_path": "smollm2-360m.pth", "tokenizer_path": "smollm2-360m-tokenizer"},
    {"name": "huggingface/virtuoso-small", "model_path": "virtuoso-small.pth", "tokenizer_path": "virtuoso-small-tokenizer"},
    {"name": "microsoft/Phi-3-mini-4k-instruct", "model_path": "phi-3-mini-4k.pth", "tokenizer_path": "phi-3-mini-4k-tokenizer"}
]

# Function to download and save the model and tokenizer
def download_and_save_model(model_name, model_path, tokenizer_path):
    print(f"Downloading model: {model_name}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        # Save the model and tokenizer locally
        print(f"Saving model to {model_path}...")
        torch.save(model.state_dict(), model_path)
        print(f"Saving tokenizer to {tokenizer_path}...")
        tokenizer.save_pretrained(tokenizer_path)
        print(f"Successfully saved {model_name}!\n")
    except Exception as e:
        print(f"Failed to download {model_name}. Error: {e}\n")

# Iterate over the models and download/save each one
for model_info in models_to_download:
    download_and_save_model(model_info["name"], model_info["model_path"], model_info["tokenizer_path"])

print("All models processed!")