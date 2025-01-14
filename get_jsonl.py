from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor
import os
import sys
import json
        
def save_dataset_as_jsonl_concurrent(dataset_name, config_name, output_file):
    """
    Downloads a dataset from Hugging Face and saves it as a JSON Lines (jsonl) file using concurrent processing.

    Args:
        dataset_name (str): Name of the dataset on Hugging Face.
        config_name (str): Configuration name of the dataset (e.g., "wikitext-2-raw-v1").
        output_file (str): Path to the output JSONL file.
    """

    # Load the dataset
    dataset = load_dataset(dataset_name, config_name)

    def save_split(split, data):
        split_output_file = output_file.replace(".jsonl", f"_{split}.jsonl")
        with open(split_output_file, "w", encoding="utf-8") as f:
            for example in data:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
        print(f"Saved {split} split to {split_output_file}")

    # Use ThreadPoolExecutor to process splits concurrently
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(save_split, split, data) for split, data in dataset.items()]
        for future in futures:
            future.result()


# Example usage
if __name__ == "__main__":
    # Dataset name and configuration
    dataset_name = sys.argv[1]

    # Output JSONL file
    os.makedirs('./data', exist_ok=True)
    output_file = f"./data/{sys.argv[1].split('/')[1]}.jsonl"
    # os.makedirs(output_file, exist_ok=True)

    # Download and save the dataset
    save_dataset_as_jsonl_concurrent(dataset_name, None, output_file)