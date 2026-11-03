import os
import csv
import time
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

# -------------------------
# Load API Key from .env
# -------------------------
load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"  # DeepSeek API endpoint
)

# -------------------------
# Read Prompts from CSV
# -------------------------
def read_prompts_from_csv(csv_file):
    prompts = []
    with open(csv_file, 'r', encoding='utf-8') as f:
        # Use csv.reader with proper quoting
        reader = csv.DictReader(f, quoting=csv.QUOTE_ALL)
        for row in reader:
            prompts.append({
                'prompt_id': row['Prompt_ID'],
                'prompt': row['evaluation_prompt']
            })
    return prompts

# -------------------------
# Process Prompts with DeepSeek
# -------------------------
def process_prompts(prompts):
    results = []
    total = len(prompts)
    
    for idx, item in enumerate(prompts, 1):
        prompt_id = item['prompt_id']
        prompt_text = item['prompt']
        
        print(f"Processing {idx}/{total}: {prompt_id}...")
        
        # Add system instruction to prevent hallucination
        full_prompt = f"""You are a supply chain analyst. Answer the following question based ONLY on the information provided. If the information is insufficient, state that you cannot determine the answer. Do not fabricate or assume information.

Question: {prompt_text}"""
        
        try:
            # Measure latency
            start_time = time.time()
            
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {
                        "role": "user",
                        "content": full_prompt
                    }
                ],
                temperature=0
            )
            
            end_time = time.time()
            latency = round(end_time - start_time, 2)
            
            output = response.choices[0].message.content
            
            results.append({
                'prompt_id': prompt_id,
                'prompt': prompt_text,
                'vanilla_gpt_output': output,
                'latency_seconds': latency
            })
            
            print(f"  ✓ Completed in {latency}s")
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            results.append({
                'prompt_id': prompt_id,
                'prompt': prompt_text,
                'vanilla_gpt_output': f"ERROR: {str(e)}",
                'latency_seconds': 0
            })
    
    return results

# -------------------------
# Save Results to CSV
# -------------------------
def save_results_to_csv(results, output_file):
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        fieldnames = ['Prompt_ID', 'Prompt', 'Vanilla_GPT_Output', 'Latency_Seconds']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in results:
            writer.writerow({
                'Prompt_ID': row['prompt_id'],
                'Prompt': row['prompt'],
                'Vanilla_GPT_Output': row['vanilla_gpt_output'],
                'Latency_Seconds': row['latency_seconds']
            })

# -------------------------
# Main Execution
# -------------------------
def main():
    # Input CSV file (update path as needed)
    input_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "initial_tests", "numerical.csv")
    
    # Check if file exists
    if not os.path.exists(input_csv):
        print(f"Error: {input_csv} not found!")
        return
    
    # Create outputs directory
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs"), exist_ok=True)
    
    # Read prompts from CSV
    print(f"Reading prompts from {input_csv}...")
    prompts = read_prompts_from_csv(input_csv)
    print(f"Found {len(prompts)} prompts to process\n")
    
    # Process all prompts
    results = process_prompts(prompts)
    
    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", f"deepseek_results_{timestamp}.csv")
    
    # Save results
    save_results_to_csv(results, output_file)
    
    print(f"\n✅ Results saved to: {output_file}")
    
    # Print summary
    total = len(results)
    successful = sum(1 for r in results if not r['vanilla_gpt_output'].startswith('ERROR'))
    avg_latency = sum(r['latency_seconds'] for r in results) / total if total > 0 else 0
    
    print(f"\n--- SUMMARY ---")
    print(f"Total prompts processed: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {total - successful}")
    print(f"Average latency: {avg_latency:.2f}s")

if __name__ == "__main__":
    main()