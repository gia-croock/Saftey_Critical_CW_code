import os
import json
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import math


def calculate_perplexity(model_id, json_file_path, text_key="response", max_length=None, stride=512):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # [1] Local model support: skip HuggingFace Hub lookup if model_id is a local directory
    local = os.path.isdir(model_id)
    print(f"Loading model: {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        local_files_only=local,
        torch_dtype=torch.bfloat16,  # [2] Memory: halves VRAM vs float32; bfloat16 avoids overflow issues common with float16
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=local)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # [3] Robustness: some models (e.g. LLaMA) have no pad token by default
    model.eval()

    if max_length is None:
        max_length = getattr(model.config, "max_position_embeddings",
                    getattr(model.config, "max_seq_len",
                    getattr(model.config, "max_sequence_length", 2048)))
    print(f"Using max_length: {max_length}")

    prompts = []
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                if text_key in item and isinstance(item[text_key], str):
                    text = item[text_key].strip().strip("\u200b\u200c\u200d\ufeff\xa0")
                    if text:
                        prompts.append(text)
                    else:
                        prompts.append(None)  # placeholder for empty responses
                else:
                    prompts.append(None)
                    print(f"Warning: Item did not contain a valid string for key '{text_key}': {item}")
    except FileNotFoundError:
        print(f"Error: The file '{json_file_path}' was not found.")
        return None, []
    except json.JSONDecodeError:
        print(f"Error: The file '{json_file_path}' is not a valid JSON file.")
        return None, []

    print(f"Successfully loaded {len(prompts)} texts from the JSON file.")

    perplexities = []

    for text in tqdm(prompts, desc="Calculating Perplexity"):

        if text is None:
            perplexities.append(None)
            continue

        encodings = tokenizer(text, return_tensors="pt")

        seq_len = encodings.input_ids.size(1)

        nlls = []
        prev_end_loc = 0

        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc

            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()

            target_ids[:, :-trg_len] = -100
            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss
            nlls.append(neg_log_likelihood)
            prev_end_loc = end_loc
            if end_loc == seq_len:
                break
        ppl = torch.exp(torch.stack(nlls).mean())
        ppl_value = ppl.item()
        if not math.isnan(ppl_value) and not math.isinf(ppl_value):
            perplexities.append(ppl_value)
        else:
            print(f"Warning: Skipped invalid perplexity value ({ppl_value}) for text: {text[:50]}...")

    valid = [p for p in perplexities if p is not None]
    if valid:
        average_perplexity = sum(valid) / len(valid)
        return average_perplexity, perplexities
    else:
        return None, perplexities


def main():
    parser = argparse.ArgumentParser(description="Calculate perplexity for texts in a JSON file using a language model")

    parser.add_argument(
        "--model_id",
        type=str,
        default="gpt2",
        help="Model ID or local path to a causal LM (e.g. gpt2, or a local LLaMA path)"
    )

    parser.add_argument(
        "--json_file",
        type=str,
        required=True,
        default=None,
        help="Path to the JSON file containing texts"
    )

    parser.add_argument(
        "--text_key",
        type=str,
        default="response",
        help="Key name for text field in JSON"
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="Maximum sequence length (auto-detected from model config if not set)"
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=512,
        help="Stride for sliding window"
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="If set, write per-item PPL scores back into this JSON file (in-place update)"  # [4] Output: annotates each JSON item with its PPL score for downstream analysis
    )

    args = parser.parse_args()

    average_perplexity, perplexities = calculate_perplexity(
        model_id=args.model_id,
        json_file_path=args.json_file,
        text_key=args.text_key,
        max_length=args.max_length,
        stride=args.stride
    )
    

    if average_perplexity is not None:
        print("\n--- Results ---")
        print(f"Total number of texts evaluated: {len(perplexities)}")
        print(f"Average Perplexity: {average_perplexity:.4f}")
    else:
        print("No valid texts were found to calculate perplexity.")
# write per-item PPL scores back to the JSON file
    if args.output_json and perplexities:
        if not os.path.exists(args.output_json):
            print(f"❌ output_json not found: {args.output_json}")
        else:
            with open(args.output_json, "r", encoding="utf-8") as f:
                data = json.load(f)

            ppl_iter = iter(perplexities)
            for item in data:
                if item.get(args.text_key, "").strip():
                    item["perplexity"] = next(ppl_iter, None)

            if average_perplexity is not None:
                data_meta = {"average_perplexity": round(average_perplexity, 4)}
                # Store summary on first item for easy access
                data[0]["average_perplexity"] = data_meta["average_perplexity"]

            with open(args.output_json, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            print(f"✅ Per-item PPL written to: {args.output_json}")


if __name__ == "__main__":
    main()