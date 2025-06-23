#!/usr/bin/env python3
"""
DeepSeek 7B Model Loader for Ubuntu Server
Loads and runs DeepSeek-Coder-7B-Instruct-v1.5 from Hugging Face
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc
import psutil
import sys
import os

def check_system_requirements():
    """Check if system has enough resources"""
    # Check RAM (needs ~14GB for 7B model)
    ram_gb = psutil.virtual_memory().total / (1024**3)
    print(f"Available RAM: {ram_gb:.1f} GB")
    
    if ram_gb < 12:
        print("WARNING: Less than 12GB RAM detected. Model may not load properly.")
    
    # Check GPU
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        print("No GPU detected - will use CPU (slower)")

def load_deepseek_model():
    """Load DeepSeek 7B model from Hugging Face"""
    model_name = "deepseek-ai/deepseek-coder-7b-instruct-v1.5"
    
    print(f"Loading model: {model_name}")
    print("This may take several minutes on first run...")
    
    try:
        # Load tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Determine device and load model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading model on: {device}")
        
        if device == "cuda":
            # Use GPU with half precision to save memory
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            model = model.to(device)
        else:
            # CPU loading
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
        
        print("Model loaded successfully!")
        return model, tokenizer, device
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure you have enough memory and stable internet connection")
        sys.exit(1)

def generate_response(model, tokenizer, device, prompt, max_length=512):
    """Generate response from the model"""
    # Format prompt for DeepSeek
    formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"
    
    # Tokenize
    inputs = tokenizer.encode(formatted_prompt, return_tensors="pt")
    if device == "cuda":
        inputs = inputs.to(device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=max_length,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the generated part
    response_start = response.find("### Response:\n") + len("### Response:\n")
    if response_start > len("### Response:\n") - 1:
        return response[response_start:].strip()
    else:
        return response.strip()

def interactive_chat(model, tokenizer, device):
    """Interactive chat loop"""
    print("\n" + "="*50)
    print("DeepSeek 7B Interactive Chat")
    print("Type 'exit' to quit, 'clear' to clear memory")
    print("="*50 + "\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() == 'exit':
                print("Goodbye!")
                break
            elif user_input.lower() == 'clear':
                # Clear GPU cache if using CUDA
                if device == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()
                print("Memory cleared!")
                continue
            elif not user_input:
                continue
            
            print("DeepSeek: ", end="", flush=True)
            response = generate_response(model, tokenizer, device, user_input)
            print(response)
            print()
            
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"Error generating response: {e}")

def main():
    """Main function"""
    print("DeepSeek 7B Model Loader")
    print("=" * 30)
    
    # Check system
    check_system_requirements()
    print()
    
    # Load model
    model, tokenizer, device = load_deepseek_model()
    
    # Test with a simple prompt
    print("\nTesting model with sample prompt...")
    test_response = generate_response(
        model, tokenizer, device, 
        "Write a Python function to calculate fibonacci numbers"
    )
    print("Test response:")
    print(test_response)
    print()
    
    # Start interactive chat
    interactive_chat(model, tokenizer, device)
    
    # Cleanup
    print("Cleaning up...")
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    # Install required packages if not available
    try:
        import transformers
        import torch
        import psutil
    except ImportError as e:
        print(f"Missing required package: {e}")
        print("Install with: pip install torch transformers psutil accelerate")
        sys.exit(1)
    
    main()