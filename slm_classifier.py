import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from termcolor import colored
import json
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from transformers import T5ForConditionalGeneration, T5Tokenizer

# # Define paths
# model_weights_path = "phi-3-mini-4k-instruct.pth"  # Update to your model path
# tokenizer_path = "phi-3-mini-4k-tokenizer"

# # Load tokenizer and model
# print("Loading tokenizer and model...")
# tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct", trust_remote_code=True)

# # Load model weights
# print("Loading model weights...")
# state_dict = torch.load(model_weights_path, map_location=torch.device("cpu"))  # Use "cuda" for GPU
# model.load_state_dict(state_dict)

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

# Load navigation functions data
with open('navigation_functions.json') as f:
    data = json.load(f)

categories = list(data.keys())
print("Categories loaded:", categories)

# Define system prompt template
def generate_prompt(user_input):
    system_prompt = (
        "You are a classification agent. Classify the following input into one of these categories:\n"
        "find_current_location, calculate_fastest_route, navigate_to_nearest_gas_station, display_real_time_traffic, search_nearby_restaurants.\n\n"
        "Output only the category name from the list above. No other words or functions can be used. No explanations or extra text.\n\n"
    )
    return system_prompt + f"Input: \"{user_input}\"\n"

# Initialize counters for accuracy
correct = 0
total = 0

# Test the model on the data
for category, prompts in data.items():
    for prompt in prompts:
        full_prompt = generate_prompt(prompt)
        
        # Debugging and user feedback
        print(colored(f"\nFull prompt:\n{full_prompt}", "yellow", attrs=["bold"]))
        
        # Tokenize and generate response
        inputs = tokenizer(full_prompt, return_tensors="pt")
        outputs = model.generate(
            **inputs,
            max_length=150,  # Limit the maximum length to prevent verbose outputs
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,  # Ensure proper padding
            no_repeat_ngram_size=2,  # Prevent repetition
            early_stopping=True  # Stop generation as soon as the model is confident
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        # Extract only the category name
        response = response.split("\n")[-1].strip()

        print(colored(f"Model response: {response}", "green", attrs=["bold"]))
        # Print response and true category
        print(colored(f"Model response: {response}", "green", attrs=["bold"]))
        print(colored(f"True Category: {category}", "magenta", attrs=["bold"]))
        
        # Update accuracy metrics
        if category.lower() in response.lower():
            correct += 1
        total += 1

# Calculate and display accuracy
accuracy = correct / total if total > 0 else 0
print(colored(f"\nAccuracy: {accuracy:.2f}", "blue", attrs=["bold"]))