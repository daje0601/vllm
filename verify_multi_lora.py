
import os
import sys
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

def verify_multi_lora():
    print("Initializing LLM with Whisper3-Turbo...")
    # Use a small whisper model for testing if turbo is too large, or just use the name provided by user
    # The user mentioned "whisper3-turbo".
    # We'll use "openai/whisper-large-v3-turbo" if it exists, or just "openai/whisper-tiny" for quick test if allowed.
    # But user specifically said "whisper3-turbo".
    model_name = "openai/whisper-large-v3-turbo" 
    
    try:
        llm = LLM(
            model=model_name,
            enable_lora=True,
            max_lora_rank=64,
            max_model_len=448, # Adjusted to match model's max_target_positions
            enforce_eager=True, # Often needed for debugging
            trust_remote_code=True
        )
        print("LLM initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize LLM: {e}")
        return

    print("Attempting to load LoRA adapters...")
    
    # Dummy LoRA paths
    lora_path_1 = "/tmp/dummy_lora_1"
    lora_path_2 = "/tmp/dummy_lora_2"
    
    # We expect this to fail with "file not found" or similar, but it confirms the request was processed.
    try:
        print(f"Sending request with LoRA 1: {lora_path_1}")
        llm.generate(
            "Hello",
            sampling_params=SamplingParams(temperature=0),
            lora_request=LoRARequest("lora1", 1, lora_path_1)
        )
    except Exception as e:
        print(f"Caught expected exception for LoRA 1: {e}")
        if "No such file or directory" in str(e) or "Error loading" in str(e):
            print("LoRA 1 loading attempt confirmed.")
        else:
            print("Unexpected error for LoRA 1.")

    try:
        print(f"Sending request with LoRA 2: {lora_path_2}")
        llm.generate(
            "Hello",
            sampling_params=SamplingParams(temperature=0),
            lora_request=LoRARequest("lora2", 2, lora_path_2)
        )
    except Exception as e:
        print(f"Caught expected exception for LoRA 2: {e}")
        if "No such file or directory" in str(e) or "Error loading" in str(e):
            print("LoRA 2 loading attempt confirmed.")
        else:
            print("Unexpected error for LoRA 2.")

if __name__ == "__main__":
    verify_multi_lora()
