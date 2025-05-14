import datasets

def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_docs(doc):        
        def remove_prefix(choice):
            return choice.split(". ", 1)[1] if ". " in choice else choice

        def format_example(doc, keys):
            question = doc["Question"].strip()
            
            choices = "".join(
                [f"{key}. {remove_prefix(choice)}\n" for key, choice in zip(keys, doc["Choices"])]
                
            )
            prompt = f"\n\nسؤال: {question}\n{choices} \nاجابة:"
            return prompt

        keys = ["A", "B", "C", "D", "E", "F"][0:len(doc["Choices"])]
        out_doc = {
            "Query":  format_example(doc, keys), 
            "Choices": keys,
            "gold": int(doc["Answer"])-1, ## 
        }        
        return out_doc
    
    return dataset.map(_process_docs)
