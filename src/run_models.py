import json
import os
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Initialize OpenAI client pointed at NVIDIA's API endpoint
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY")
)

# Configuration for all target models and their required extra parameters
MODELS_TO_TEST = [
    {
        "name": "GLM 5.2",
        "slug": "z-ai/glm-5.2",
        "extra_body": None,
        "max_tokens": 2048
    },
    {
        "name": "Nemotron 3 Ultra (550B)",
        "slug": "nvidia/nemotron-3-ultra-550b-a55b",
        "extra_body": {"chat_template_kwargs": {"enable_thinking": True}, "reasoning_budget": 4096},
        "max_tokens": 4096
    },
    {
        "name": "Kimi k2.6",
        "slug": "moonshotai/kimi-k2.6",
        "extra_body": None,
        "max_tokens": 2048
    },
    {
        "name": "DeepSeek V4 Flash",
        "slug": "deepseek-ai/deepseek-v4-flash",
        "extra_body": {"chat_template_kwargs": {"thinking": True, "reasoning_effort": "high"}},
        "max_tokens": 4096
    },
    {
        "name": "Gemma 4 31B IT",
        "slug": "google/gemma-4-31b-it",
        "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
        "max_tokens": 4096
    },
    {
        "name": "Nemotron 3 Nano (30B)",
        "slug": "nvidia/nemotron-3-nano-30b-a3b",
        "extra_body": {"reasoning_budget": 4096},
        "max_tokens": 4096
    },
    {
        "name": "GPT-OSS 120B",
        "slug": "openai/gpt-oss-120b",
        "extra_body": None,
        "max_tokens": 2048
    },
    {
        "name": "Llama 3.3 70B Instruct",
        "slug": "meta/llama-3.3-70b-instruct",
        "extra_body": None,
        "max_tokens": 1024
    }
]

# Load prompts
with open("data/prompts.json", "r") as f:
    prompts = json.load(f)

results = []

print(
    f"Starting evaluations for {len(MODELS_TO_TEST)} models across {len(prompts)} prompts...")
print("Pacing requests to respect 40 RPM limit (2s delay per call)...\n")

for model_config in MODELS_TO_TEST:
    model_slug = model_config["slug"]
    model_name = model_config["name"]
    print(f"--- Testing Model: {model_name} ({model_slug}) ---")

    for prompt_item in prompts:
        prompt_id = prompt_item["id"]
        category = prompt_item["category"]
        user_prompt = prompt_item["prompt"]

        try:
            # Build API parameters
            kwargs = {
                "model": model_slug,
                "messages": [{"role": "user", "content": user_prompt}],
                "temperature": 0.7,
                "max_tokens": model_config["max_tokens"],
                "stream": False
            }

            if model_config["extra_body"]:
                kwargs["extra_body"] = model_config["extra_body"]

            response = client.chat.completions.create(**kwargs)
            message = response.choices[0].message

            # Extract reasoning/thinking content if generated
            reasoning = getattr(message, "reasoning_content", None) or getattr(
                message, "reasoning", None)
            content = message.content or ""

            results.append({
                "id": prompt_id,
                "category": category,
                "prompt": user_prompt,
                "model_tested": model_slug,
                "model_name": model_name,
                "reasoning": reasoning,
                "response": content
            })

            print(f"  [SUCCESS] Prompt '{prompt_id}' processed.")

        except Exception as e:
            print(f"  [ERROR] Prompt '{prompt_id}' failed: {e}")

        # Rate limiting: 2 seconds delay keeps speed well within the 40 RPM limit
        time.sleep(2)

    print()

# Save final benchmark output
with open("data/results.json", "w") as f:
    json.dump(results, f, indent=4)

print("=" * 50)
print(f"Execution complete! Total responses collected: {len(results)}")
print("Results saved to 'data/results.json'")
