import os
import argparse
from datasets import load_dataset

def main(args):
    dataset = load_dataset('json', data_files=args.data_file)
    dataset = dataset['train']
    text_list = []
    for i in range(len(dataset)): 
        text = f"Instruct: {dataset[i]['input']}\nOutput: {dataset[i]['answer_icliniq']}<|endoftext|>"
        text_list.append(text)
    dataset = dataset.add_column("text", text_list)
    dataset.to_json("iCliniq_full.jsonl")


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, default='iCliniq.json', help="json file containing iCliniq dataset")
    args = parser.parse_args()
    main(args)
