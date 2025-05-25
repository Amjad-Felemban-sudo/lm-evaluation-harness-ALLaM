import datasets

def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_docs(doc):
        def format_example(doc, keys):
            question =  doc["question"].strip()
            
            choices = "".join(
                [f"{key}. {choice}\n" for key, choice in zip(keys,  doc["choices"])]
            )
            prompt = f"السؤال: {question}\n{choices}\nالاجابة:"
            return prompt

        keys_ar = ["أ", "ب", "ج", "د"]
        keys_en = ["A", "B", "C", "D"]
        out_doc = {
                "query": format_example(doc, keys_en),
            "choices": keys_en,
            "gold": int(doc["label"]) - 1,
        }
        return out_doc
    
    return dataset.map(_process_docs)



