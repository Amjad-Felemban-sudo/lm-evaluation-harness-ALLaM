import datasets
import re
from typing import Dict, List

def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_docs(doc):
        def remove_prefix(choice):
            prefixes = ["(A)", "(B)", "(C)", "(D)"]
            for prefix in prefixes:
                if choice.startswith(prefix + " "):
                    return choice[len(prefix) + 1:]  
            return choice 

        def format_example(doc, keys):
            question = doc["question"].strip()
            choices = "".join(
                [f"{key}. {remove_prefix(choice)}\n" for key, choice in zip(keys, doc["options"])]
            )

            prompt = f"\n\nالسؤال: {question}\n{choices}\nالاجابة:"
            return prompt

        keys_en = ["A", "B", "C", "D"]
        out_doc = {
            "query": format_example(doc, keys_en),
            "choices": keys_en,
            "gold": keys_en.index(doc["label"]),
        }
        return out_doc
    
    return dataset.map(_process_docs)

def extract_boxed_text(text, gold):
    pattern = r'boxed{(.*?)}|framebox{(.*?)}|<answer>(.*?)</answer>|<\|begin_of_solution\|> {(.*?)} <\|end_of_solution\|>'
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        for match in matches[::-1]:
            for group in match:
                if group != "":
                    return group
    pattern = r'\d+'  # get the last integer if no pattern found
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[-1]
    elif gold in text:
        return gold
    return ""

def process_results(doc: dict, results: List[str]) -> Dict[str, int]:
    response = results[0]
    pred = extract_boxed_text(response, gold)
    gold = doc["answer"]
    results = {
        "acc": 1 if pred == gold else 0
    }
    return results