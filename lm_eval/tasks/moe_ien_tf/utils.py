import datasets

def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_docs(doc):
        keys=["صحيحة",
              "خاطئة"
              ]
        #keys =["صواب",
        #         "خطأ"]
        target_key = int(doc["Answer"])-1

        out_doc = {
            "query":  "\n\nالسؤال:" +doc["Question"]+"\nإجابة:'", 
            "choices": keys,
            "gold": target_key,
        }
        return out_doc
    return dataset.map(_process_docs)

# import datasets


# def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
#     def _process_docs(doc):
#         def remove_prefix(choice):
#             choices_name = {
#                 'صواب': 'صحيحة',
#                 'خطأ': 'خاطئة',
#             }
#             if choice[0].isdigit():
#                 prefix = choice[2:].strip()
            
            
#             return [choices_name[prefix]]

#         # def remove_prefix(choice):
#         #     #prefixes = ["صح", "خطأ"]
#         #     for prefix in prefixes:
#         #         choice = choice.replace('صواب', 'صحيحة').replace('خطأ','خاطئة')
#         #         #if choice.startswith(prefix + " "):
#         #         #    return choice[len(prefix) + 1:]  
#         #         if choice[0].isdigit():
#         #             return choice[len(prefix) + 1: ]

#         #     return choice 

#         def format_example(doc, keys):
#             question = doc["Question"].strip()
#             choices = "".join(
#                 [f"{key}. {remove_prefix(choice)}\n" for key, choice in zip(keys, doc["Choices"])]
#             )
#             print(question)
#             print(choices)
#             prompt = f"السؤال: {question}\nإجابة:\n{choices}"
#             return prompt
#         keys_en = ["1","2"]
#         print(keys_en.index(doc["Answer"]))
#         gold = keys_en.index(doc["Answer"])
#         print(["خاطئة","صحيحة"][gold])
#         assert False

#         out_doc = {
#             "query": format_example(doc, keys_en),
#             "choices": ["خاطئة","صحيحة"],
#             "gold": keys_en.index(doc["Answer"]),
#             "Speciality":doc['Speciality']
#         }
    
#         # print(f"doc: {doc}")
#         # print(f"out_doc: {out_doc}")
#         #append_to_json_lines('/home/khalid/Documents/evaluation/eval_inputs/evalharness_new_moe_ien.jsonl',out_doc)


#         return out_doc
    
#     return dataset.map(_process_docs)


