# coding=utf-8
# https://github.com/huggingface/datasets/blob/main/templates/new_dataset_script.py
# Lint as: python3
"""The MOE-IEN-TF datasey."""


import csv
import os
import textwrap

import numpy as np
import pandas as pd 
import datasets
import sys
import ast
dirname = os.getcwd()
print(dirname)
# DATA_ROOT = dirname+"/lm-evaluation-harness/lm_eval/datasets/moe_ien/eval_set_v4/"
#DATA_ROOT = dirname+"/lm_eval/datasets/moe_ien/eval_set_v4/"
dirname = os.getcwd()
DATA_ROOT = os.path.join(dirname, "lm_eval", "tasks", "moe_ien_tf/")

_MOE_IEN_TF_CITATION = """\
@inproceedings{
    
}
"""

_MOE_IEN_TF_DESCRIPTION = """\


"""


class MOE_IEN_TFConfig(datasets.BuilderConfig):
    """BuilderConfig for MOE_IEN_TF."""

    def __init__(
        self,
        text_features,
        label_column,
        data_url,
        data_dir,
        citation,
        url,
        label_classes=None,
        process_label=lambda x: str(x),
        **kwargs,
    ):
        super(MOE_IEN_TFConfig, self).__init__(version=datasets.Version("1.0.0", ""), **kwargs)
        self.text_features = text_features
        self.label_column = label_column
        self.label_classes = label_classes
        self.data_url = data_url
        self.data_dir = data_dir
        self.citation = citation
        self.url = url
        self.process_label = process_label


class MOE_IEN_TF_SS(datasets.GeneratorBasedBuilder):
    """The MOE_IEN_TF benchmark."""

    BUILDER_CONFIGS = [
        MOE_IEN_TFConfig(
            name= "moe_ien_tf",
            description=textwrap.dedent("IN ARABIC "),
            text_features={"Question": "Question",
                                 "QuestionCode": 'QuestionCode',
                                "Subject": "Subject",
                                "Stage": "Stage",
                                "Grade": "Grade",
                                "domain": "domain",
                               "DifficultyFactor": "DifficultyFactor",
                          }, 
            label_classes=["1", "2"],
            label_column="Answer",
            data_url="",
            data_dir="",
            citation=textwrap.dedent(
                """\
            @inproceedings{???
            } 
            """
            ),
            url="",
        )
    ]

    
    
    def _info(self):
        features = {"Question": datasets.Value("string"),
                    "QuestionCode": datasets.Value("string"),
                    "Stage": datasets.Value("string"),
                    "Subject": datasets.Value("string"),
                    "Grade": datasets.Value("string"),
                    "domain": datasets.Value("string"),
                    "DifficultyFactor": datasets.Value("string"),
                    
                   }
        features["Answer"] = datasets.Value("string")
        features["idx"] = datasets.Value("int32")
        print(self.config)
        return datasets.DatasetInfo(
            description=_MOE_IEN_TF_DESCRIPTION,
            features=datasets.Features(features),
            # homepage=self.config.url,
            # citation=self.config.citation + "\n" + _MOE_IEN_TF_CITATION,
        )

    def _split_generators(self, dl_manager):
        list_splits = []
        
        list_splits.append(
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={
                        "data_file": os.path.join(DATA_ROOT + "tf_test_balanced.xlsx"),
                        "split": 'test',
                    },
                )
            )
        list_splits.append(
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={
                        "data_file": os.path.join(DATA_ROOT + "tf_dev.xlsx"),
                        "split": 'dev',
                    },
                )
            )
          
        return list_splits
      

    def _generate_examples(self, data_file, split, mrpc_files=None):
        # process_label = self.config.process_label
        # label_classes = self.config.label_classes
        df = pd.read_excel(data_file)
        i = 0
        for n, row in df.iterrows():
            try:
                # Attempt to evaluate the expression
                answer_value = ast.literal_eval(row['Answer'])[0]
            except SyntaxError as e:
                # Handle the case where an unterminated string literal error occurs
                print("Error: Unterminated string literal detected.")
                # You might want to extract the value from the error message if needed
                error_message = str(e)
                value_start_index = error_message.find("['") + 1
                value_end_index = error_message.find("']")
                if value_start_index != -1 and value_end_index != -1:
                    answer_value = error_message[value_start_index:value_end_index]
                else:
                    # Default to None if unable to extract the value
                    answer_value = None
                    continue
            except (ValueError, IndexError) as e:
                # Handle other potential errors during evaluation
                print(f"Error occurred while evaluating 'Answer': {e},{n},{row}")
                # Assign a default value or handle the error in a way appropriate for your application
                answer_value = None
                continue
            row = {
                    "QuestionCode": row['QuestionCode'],
                    "Question": row['Question'],
                    "Subject": row['Speciality'],
                    "Stage": row['Stage'],
                    "Grade": row['Level'],
                    "DifficultyFactor": row['DifficultyFactor'],
                    "domain": f"{row['Stage']}_{row['Speciality']}",
                    "Answer": answer_value,
                }
            
            example = {feat: row[col] for feat, col in self.config.text_features.items()}
            example["idx"] = i
            example["Answer"] =row['Answer']
            i+=1
           

            # Filter out corrupted rows.
            for value in example.values():
                if value is None:
                    break
            else:
                yield example["idx"], example
