import json

Passage = "passage"
Question = "question"
Options = "options"
Equation = "equation"
Label = "label"
Answer = "answer"


input_file = "araMath_dev.jsonl"
output_file = "araMath_dev_decoded.jsonl"

with open(input_file, 'r', encoding="utf-8") as fin:
    with open(output_file, 'w', encoding="utf-8") as fout:
        for line in fin:
            data = json.loads(line)
            # print(f"""
            # {Passage}: {data[Passage]},
            # {Question}: {data[Question]},
            # {Options}: {data[Options]},
            # {Equation}: {data[Equation]},
            # {Label}: {data[Label]},
            # {Answer}: {data[Answer]}
            # """)
            fout.write(json.dumps(data, ensure_ascii=False) + '\n')