import os
import time
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from the .env file
load_dotenv()

# Initialize the client. It will automatically look for OPENAI_API_KEY in the environment.
client = OpenAI(base_url="http://192.168.1.102:1234/v1", api_key="local")

def benchmark_token_speed():
    # A prompt designed to generate a relatively long response for a better speed average
    conversation_history = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Please write a detailed 400-word essay about the history of the Colosseum in Rome."}
    ]

    print("Sending request to OpenAI...\n")

    start_time = time.time()
    first_token_time = None
    
    try:
        # We use stream=True to measure generation speed as it happens.
        # stream_options={"include_usage": True} is required to get exact token counts at the end of the stream.
        
        response = client.chat.completions.create(
            model="qwen3-coder-30b-a3b-instruct",
            messages=conversation_history,
        )
                
        completion_tokens = 0
        
        for chunk in response:
            # Mark the exact time the first content chunk arrives (Time to First Token)
            if first_token_time is None and chunk.choices and chunk.choices[0].delta.content:
                first_token_time = time.time()
                print("✅ Connected! First token received. Generating response...")
                
            # The final chunk contains the exact usage statistics
            if chunk.usage:
                completion_tokens = chunk.usage.completion_tokens

    except Exception as e:
        print(f"Error during API call: {e}")
        return

    end_time = time.time()
    
    # Calculate metrics
    ttft = first_token_time - start_time if first_token_time else 0
    
    # Generation time is the time spent strictly generating tokens (ignoring initial connection latency)
    generation_time = end_time - first_token_time if first_token_time else 0
    total_time = end_time - start_time
    
    # Calculate final tokens per second
    tokens_per_second = completion_tokens / generation_time if generation_time > 0 else 0
    
    print("\n" + "-" * 40)
    print("📊 Benchmark Results:")
    print(f"Time to First Token (TTFT): {ttft:.3f} seconds")
    print(f"Generation Time:            {generation_time:.3f} seconds")
    print(f"Total Time Elapsed:         {total_time:.3f} seconds")
    print(f"Total Completion Tokens:    {completion_tokens} tokens")
    print(f"Token Generation Speed:     {tokens_per_second:.1f} tokens/second")

if __name__ == "__main__":
    benchmark_token_speed()