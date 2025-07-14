import os
import sys
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.config_manager import load_config

config = load_config()

def generate_jinja2_template(template_data):
    """
    Converts template data into a Jinja2 chat template string.
    """
    system_prefix = template_data.get("system_prompt", {}).get("prefix", "")
    system_suffix = template_data.get("system_prompt", {}).get("suffix", "")
    user_prefix = template_data.get("user_turn", {}).get("prefix", "")
    user_suffix = template_data.get("user_turn", {}).get("suffix", "")
    assistant_prefix = template_data.get("assistant_turn", {}).get("prefix", "")
    assistant_suffix = template_data.get("assistant_turn", {}).get("suffix", "")
    
    jinja_template = f"""{{%- for message in messages %}}
    {{%- if message['role'] == 'system' %}}
{system_prefix}{{{{ message['content'] }}}}{system_suffix}
    {{%- elif message['role'] == 'user' %}}
{user_prefix}{{{{ message['content'] }}}}{user_suffix}
    {{%- elif message['role'] == 'assistant' %}}
{assistant_prefix}{{{{ message['content'] }}}}{assistant_suffix}
    {{%- endif %}}
{{%- endfor %}}
{{%- if add_generation_prompt %}}
{assistant_prefix}
{{%- endif %}}"""
    
    return jinja_template.strip()

def update_tokenizer_config(target_path, template_data, jinja_template):
    """
    Updates the tokenizer_config.json with chat template and special tokens.
    """
    tokenizer_config_path = os.path.join(target_path, "tokenizer_config.json")
    
    # Load existing tokenizer config
    try:
        with open(tokenizer_config_path, 'r', encoding='utf-8') as f:
            tokenizer_config = json.load(f)
    except FileNotFoundError:
        tokenizer_config = {}
    
    # Add chat template
    tokenizer_config["chat_template"] = jinja_template
    
    # Override bos_token and eos_token if specified
    if "bos_token" in template_data and template_data["bos_token"]:
        tokenizer_config["bos_token"] = template_data["bos_token"]
    
    if "eos_token" in template_data and template_data["eos_token"]:
        tokenizer_config["eos_token"] = template_data["eos_token"]
    
    # Save updated tokenizer config
    with open(tokenizer_config_path, 'w', encoding='utf-8') as f:
        json.dump(tokenizer_config, f, indent=2, ensure_ascii=False)
    
    print(f"Updated tokenizer_config.json with chat template and special tokens")

def process_model():
    """
    Loads a base model, adds special tokens from a template,
    and saves the extended model and tokenizer to a new location.
    """
    # --- 1. Load Configuration ---
    # Access config attributes through config.source
    source_path = config.source.model.source_model_path + "/" + config.source.model.model_name
    target_path = config.source.model.extended_model_path + "/" + config.source.model.model_name
    template_path = config.source.template.path

    print(f"Source model path: {source_path}")
    print(f"Target model path: {target_path}")
    print(f"Template path: {template_path}")

    # --- 2. Load Model and Tokenizer from Source ---
    print("Loading base model and tokenizer from source...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(source_path)
        model = AutoModelForCausalLM.from_pretrained(
            source_path,
            # Corrected access to torch_dtype
            torch_dtype=getattr(torch, config.source.model.torch_dtype, torch.float32)
        )
    except Exception as e:
        print(f"Error loading model/tokenizer from {source_path}: {e}")
        return

    # --- 3. Add Special Tokens ---
    print("Adding special tokens...")
    try:
        with open(template_path, 'r') as f:
            template = json.load(f)
        added_tokens = template.get("added_tokens", [])

        if not added_tokens:
            print("No new tokens to add.")
        else:
            # Add new tokens to the tokenizer
            tokenizer.add_special_tokens({"additional_special_tokens": added_tokens})

            # Resize the model's token embeddings
            model.resize_token_embeddings(len(tokenizer))

            # Initialize new embeddings randomly (default behavior)
            print(f"Added {len(added_tokens)} new tokens: {added_tokens}")

    except FileNotFoundError:
        print(f"Error: Template file not found at {template_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {template_path}")
        return

    # --- 4. Save Updated Model and Tokenizer to Target ---
    print(f"Saving updated model and tokenizer to {target_path}...")
    try:
        os.makedirs(target_path, exist_ok=True)
        model.save_pretrained(target_path)
        tokenizer.save_pretrained(target_path)
        print("Model processing complete!")
    except Exception as e:
        print(f"Error saving model/tokenizer to {target_path}: {e}")
        return

    # --- 5. Generate Jinja2 Template ---
    print("Generating Jinja2 chat template...")
    try:
        jinja_template = generate_jinja2_template(template)
        print("Jinja2 template generated successfully")
    except Exception as e:
        print(f"Error generating Jinja2 template: {e}")
        return

    # --- 6. Update tokenizer_config.json with Chat Template and Special Tokens ---
    print("Updating tokenizer_config.json...")
    try:
        update_tokenizer_config(target_path, template, jinja_template)
        print("tokenizer_config.json updated successfully")
    except Exception as e:
        print(f"Error updating tokenizer_config.json: {e}")
        return

    print("All processing steps completed successfully!")

if __name__ == "__main__":
    process_model()
