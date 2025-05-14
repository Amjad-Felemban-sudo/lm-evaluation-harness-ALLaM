import datasets

def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_docs(doc):        
        def remove_prefix(choice):
            return choice.replace('.', '') if '.' in choice[:2] else choice
        
        def format_example(doc, keys):
            question = doc["question"].strip()
            
            choice_num = ['choice1', 'choice2', 'choice3', 'choice4']
            choices = "".join(
                [f"{key}. {remove_prefix(doc[choice_num[index]])}\n" for index, key in enumerate(keys)]
            )

            prompt = f"\n\n\nالسؤال: {question}\n{choices} \nالاجابة:"
            return prompt

        #keys = ["1", "2", "3", "4"]
        keys = ["A", "B", "C", "D"]
        out_doc = {
            "query":  format_example(doc, keys), 
            "choices": keys,
            "gold": doc["answer"]-1,
        }        

        return out_doc
    
    return dataset.map(_process_docs)

