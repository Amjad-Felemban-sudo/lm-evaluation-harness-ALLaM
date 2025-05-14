import datasets
import numpy as np
import sacrebleu
from rouge_score import rouge_scorer, scoring
ROUGE_SCORER = None

def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _format_answers(answers):
        formatted_answers = []
        for answer in answers:
            answer = answer.strip()
            if len(answer):
                if answer[-1] != ".":
                    formatted_answers.append(answer + ".")
                else:
                    formatted_answers.append(answer)
        return formatted_answers

    def format_example(doc):
        question = doc["input"].strip()
        options = '\n'.join(doc['options'])
        prompt = f"السؤال: {question}\nالخيارات:\n{options}\nالاجابة:"
        return prompt

    def _process_doc(doc):
        letter =  "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        answer_key =  letter.index(doc['label'])
        keys = [letter[i] for i in range(len(doc['options']))]
        out_doc = {
            "query": format_example(doc),
            "choices":doc['options'],
            "gold": answer_key,
        }
        return out_doc

    return dataset.map(_process_doc)
