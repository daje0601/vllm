import os
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.assets.audio import AudioAsset

def main():
    # 1. Initialize the model with LoRA enabled
    # max_model_len is set to 448 as per Whisper's config
    llm = LLM(
        model="openai/whisper-large-v3-turbo",
        enable_lora=True,
        max_model_len=448,
        dtype="half",
        gpu_memory_utilization=0.8, # Adjust as needed
    )

    # 2. Prepare input data
    # Using a sample audio file from vLLM assets
    audio_asset = AudioAsset("mary_had_lamb")
    audio_data = audio_asset.audio_and_sample_rate
    
    # Define the prompt structure for Whisper
    # You can change the tokens to control language, task, etc.
    prompt = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>"
    
    inputs = {
        "prompt": prompt,
        "multi_modal_data": {
            "audio": audio_data,
        },
    }

    # 3. Define LoRA Request
    # IMPORTANT: Replace 'path/to/your/lora' with your actual LoRA path
    # If you don't have a LoRA yet, you can comment out the lora_request argument below to run base model inference
    lora_path = "/path/to/your/lora_adapter" 
    
    # Check if the path exists to avoid immediate error if user runs this blindly
    if not os.path.exists(lora_path):
        print(f"Warning: LoRA path '{lora_path}' does not exist.")
        print("Running without LoRA for demonstration purposes.")
        lora_request = None
    else:
        print(f"Loading LoRA from {lora_path}")
        lora_request = LoRARequest("my_whisper_lora", 1, lora_path)

    # 4. Run Inference
    sampling_params = SamplingParams(temperature=0, max_tokens=100)
    
    print("Running generation...")
    outputs = llm.generate(
        [inputs],
        sampling_params=sampling_params,
        lora_request=lora_request
    )

    # 5. Print Results
    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"Generated text: {generated_text}")

if __name__ == "__main__":
    main()
