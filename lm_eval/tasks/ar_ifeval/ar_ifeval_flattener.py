import json

INPUT_PATH = "Arabic_IF_Eval.jsonl"
OUTPUT_PATH = "Arabic_IF_Eval_Flat.jsonl"
PROMPT_KEY = "prompt"
CATEGORIES_KEY = "categories"

def flatten_record(record):
    # Starting with only top level records and ignoring multi-hierarchical items (instruction_following_prompt)
    flat = {k: v for k, v in record.items() if k != "instruction_following_prompt"}
    # Extracting and flattening nested fields/keys
    if "instruction_following_prompt" in record:
        # If key "instruction_following_prompt" exits, replicate the key at one level higher in the flat dictionary
        prompt_obj = record["instruction_following_prompt"]
        flat[PROMPT_KEY] = prompt_obj.get(PROMPT_KEY)
        flat[CATEGORIES_KEY] = prompt_obj.get(CATEGORIES_KEY)
    return flat

# Open both files (input file: infile | output file: outfile), strip each sample, flatten, and load to the output file
with open(INPUT_PATH, "r", encoding="utf-8") as infile, open(OUTPUT_PATH, "w", encoding="utf-8") as outfile:
    for line in infile:
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        flat_record = flatten_record(record)
        outfile.write(json.dumps(flat_record, ensure_ascii = False) + "\n")