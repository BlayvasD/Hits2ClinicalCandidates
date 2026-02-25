"""
LLM Batch Execute

This script loads a specified prompt and a batch of abstracts, then queries the OpenDFM 
ChemDFM-v1.5-8B model for each abstract using the prompt. The model's output is expected 
to contain a JSON object, which is extracted, augmented with the title, and saved to a 
file. The script also measures and prints the total execution time.

inputs:
1. prompt_path: Path to a text file containing the prompt to be used for querying the model.
    /nfs/exk/work/clinicalcandidates/automated_querying/prompts/analyzeAbstract.txt
2. abstracts_path: Path to a CSV file containing titles and abstracts. Titles should be in column 5 and abstracts in column 6.
    /nfs/exk/work/clinicalcandidates/automated_querying/brown_abstracts/brown_allExamples_w_abstracts.csv

outputs:
1. brown_outputs.json: A file containing the extracted JSON objects from the model's responses, one per line.
"""

print("Starting LLM batch execution...")

from sys import argv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, TextIteratorStreamer
import threading
import csv
import json
import re
from time import perf_counter, sleep

print("Imports completed.")


def load_tokenizer_and_model():
    if not torch.cuda.is_available():
        print('No CUDA device available. Exiting...')
        exit()
    model_name_or_id = "OpenDFM/ChemDFM-v1.5-8B"
    cache_path = "/nfs/exk/work/clinicalcandidates/blayvas/models"
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_id, 
        trust_remote_code=True,
        cache_dir = cache_path
        )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_id, 
        trust_remote_code=True, 
        dtype=torch.float16, 
        device_map="auto",
        cache_dir = cache_path,
        pad_token_id=tokenizer.eos_token_id)
    return tokenizer, model

def load_prompt(path):
    with open(path, 'r') as f:
        prompt = f.read()
    return(prompt)

def load_abstract_batch(infile):
    titles = []
    abstracts = []
    with open(infile, 'r') as f:
        reader = csv.reader(f)
        reader.__next__()  # skip header
        for ll in reader:
            titles.append(ll[5])
            abstracts.append(ll[6])
    return titles,abstracts

def isolate_json(text: str, title) -> str:
    """
    Extract the first valid JSON object from LLM output text,
    remove Markdown/code fences if present, add a 'title' field,
    and return as a JSON string.
    """
    # Remove triple backtick fences with optional "json"
    text = re.sub(r"```json|```", "", text, flags=re.IGNORECASE).strip()

    decoder = json.JSONDecoder()
    for start_idx, char in enumerate(text):
        if char != '{':  # only dicts
            continue
        try:
            json_obj, end_idx = decoder.raw_decode(text[start_idx:])
            if not isinstance(json_obj, dict):
                continue
            json_obj["title"] = title
            return json.dumps(json_obj, ensure_ascii=False, indent=2)
        except json.JSONDecodeError:
            continue

    print("Failed to find JSON in text:\n", text[:500])  # log first 500 chars
    raise ValueError("No valid JSON object found in text.")


def query(prompt, titles, texts, tokenizer, model, batch_size=8):
    """Run the prompt against a list of inputs in batches and return a list of
    JSON strings.

    We concatenate each abstract with the prompt, then perform a single
    `model.generate` pass for each batch.  Streaming is dropped in favour of
    the simpler batch API, which allows us to submit many examples at once and
    get much higher throughput when the CUDA device is the bottleneck.

    ``batch_size`` can be tuned depending on your GPU memory.  If any
    response fails JSON extraction we skip it and continue.
    """

    results = []
    # iterate over the data in fixed-size batches
    for i in range(0, len(texts), batch_size):
        batch_titles = titles[i : i + batch_size]
        batch_texts = texts[i : i + batch_size]

        # build the list of input strings and tokenize as a batch
        batch_inputs = [t + "\n" + prompt for t in batch_texts]
        delay = 5
        while True:
            try:
                inputs = tokenizer(
                    batch_inputs,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to("cuda")
                break  # success, exit the retry loop
            except torch.AcceleratorError:
                print("CUDA out of memory during tokenization. Retrying after a short delay...")
                torch.cuda.empty_cache()
                sleep(delay)  # wait before retrying
                delay = min(delay * 2, 60)  # exponential backoff with a max delay
        try:
            inputs = tokenizer(
                batch_inputs,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to("cuda")
        except torch.AcceleratorError:
            print("Error tokenizing batch. Skipping this batch.")
            continue

        # generate all outputs at once (no streamer)
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.2,
        )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        # strip the corresponding prompt prefix and isolate JSON
        for inp, dec, title in zip(batch_inputs, decoded, batch_titles):
            response_only = dec[len(inp) :]
            try:
                results.append(isolate_json(response_only, title))
            except ValueError:
                # log inside isolate_json already, just skip
                continue
    return results

def main():
    print("Functions defined, initiating main execution.")
    prompt_path = argv[1]
    abstracts_path = argv[2]  # Titles must be csv col 5, abstracts col 6. Not updating for now.
    prompt = load_prompt(prompt_path)
    titles,abstracts = load_abstract_batch(abstracts_path)
    tokenizer,model = load_tokenizer_and_model()

    start = perf_counter()
    # now run the entire dataset in batches rather than one prompt at a time
    all_json = query(prompt, titles, abstracts, tokenizer, model)

    # the returned list already contains only successfully parsed items
    print(all_json)
    end = perf_counter()
    print(f"Performance counter: {end - start:.6f} seconds")

    with open('brown_outputs.json','w') as f:
        f.write('[')
        for j in all_json[:-1]:
            f.write(j + ',\n')
        f.write(all_json[-1] + '\n]')


main()
