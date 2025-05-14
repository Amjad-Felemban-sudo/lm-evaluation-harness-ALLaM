# coding=utf-8
# https://github.com/huggingface/datasets/blob/main/templates/new_dataset_script.py
# Lint as: python3
""" SDAIA MCQ benchmark."""

from lm_eval.api.task import ConfigurableTask
import os

import csv
import os
import textwrap

import pandas as pd
import numpy as np

import datasets
from pathlib import Path
import ast
dirname = os.getcwd()
DATA_ROOT = os.path.join(dirname, "lm_eval", "tasks", "sdaia_mcq_ext")

_SDAIA_MCQ_CITATION = """\
@inproceedings{
    
}
"""

_SDAIA_MCQ_DESCRIPTION = """\


"""


class SdaiaMcqConfig(datasets.BuilderConfig):
    """BuilderConfig for sdaia_mcq_ext."""

    def __init__(
        self,
        text_features,
        label_column,
        data_url,
        data_dir,
        citation,
        url,
        label_classes=None,
        process_label=lambda x: x,
        **kwargs,
    ):
        """BuilderConfig for SdaiaMcqConfig.

        Args:
          text_features: `dict[string, string]`, map from the name of the feature
            dict for each text field to the name of the column in the csv file
          label_column: `string`, name of the column in the csv file corresponding
            to the label
          data_url: `string`, url to download the zip file from
          data_dir: `string`, the path to the folder containing the tsv files in the
            downloaded zip
          citation: `string`, citation for the data set
          url: `string`, url for information about the data set
          label_classes: `list[string]`, the list of classes if the label is
            categorical. If not provided, then the label will be of type
            `datasets.Value('float32')`.
          process_label: `Function[string, any]`, function  taking in the raw value
            of the label and processing it to the form required by the label feature
          **kwargs: keyword arguments forwarded to super.
        """
        super(SdaiaMcqConfig, self).__init__(**kwargs)
        self.text_features = text_features
        self.label_column = label_column
        self.label_classes = label_classes
        self.data_url = data_url
        self.data_dir = data_dir
        self.citation = citation
        self.url = url
        self.process_label = process_label



class sdaia_mcq_ext(datasets.GeneratorBasedBuilder):
    """The SDAIA_MCQ benchmark."""

    BUILDER_CONFIGS = [
        
        SdaiaMcqConfig(
                name= "sdaia_mcq_ext",
                description=textwrap.dedent("sdaia_mcq_ext."),
                text_features={"id":"id",
                               "question":"question",
                               "choice1":"choice1",
                               "choice2":"choice2",
                               "choice3":"choice3",
                               "choice4":"choice4",
                                "domain":"domain",
                              "sub-domain":"sub-domain",
                              },

                
                label_classes=["1", "2", "3", "4"],
                label_column="answer",
                data_url="",
                data_dir="",
                citation=textwrap.dedent(
                    """\
                @inproceedings{???
                } 
                """
                ),
                url="",
            ),
                    
        ]

    
    
    def _info(self):
        features = {"id": datasets.Value("int32"),
                    "question": datasets.Value("string"),
                    "choice1": datasets.Value("string"),
                    "choice2": datasets.Value("string"),
                    "choice3": datasets.Value("string"),
                    "choice4": datasets.Value("string"),
                    "domain": datasets.Value("string"),
                    "sub-domain": datasets.Value("string"),
                    # "source": datasets.Value("string"),

                }
        #features["answer"] = datasets.Value("string")
        features['answer'] = datasets.Value("int32")
        return datasets.DatasetInfo(
            description=_SDAIA_MCQ_DESCRIPTION,
            features=datasets.Features(features),
            homepage=self.config.url,
            citation=self.config.citation + "\n" + _SDAIA_MCQ_CITATION,
            
        )

    def _split_generators(self, dl_manager):
        list_splits = []
        
        list_splits.append(
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={
                        "data_file": os.path.join(DATA_ROOT , "sdaia_mcqs_test.xlsx"),
                        "split": 'test',
                    },
                )
                
            )
        list_splits.append(
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={
                        "data_file": os.path.join(DATA_ROOT , "sdaia_mcqs_dev.xlsx"),
                        "split": 'dev',
                    },
                )
        )
        print(list_splits)

        return list_splits
    
    def _generate_examples(self, data_file, split, mrpc_files=None):
        process_label = self.config.process_label
        label_classes = self.config.label_classes
        print("split====",split)
        reader = pd.read_excel(data_file)
        for n, row in reader.iterrows():
            example = {
                    "id":n,
                    "question": row["question"].strip(),
                    "choice1": str(row["choice1"]).strip(),
                    "choice2": str(row["choice2"]).strip(),
                    "choice3": str(row["choice3"]).strip(),
                    "choice4": str(row["choice4"]).strip(),
                    "answer": str(row["answer"]).strip(),
                    "domain": str(row["domain"]).strip(),
                    "sub-domain": str(row["sub-domain"]).strip(),

                }
            
            if self.config.label_column in row:
                label = row[self.config.label_column]
                if label_classes and label not in label_classes:
                    label = int(label) if label else None
                example["answer"] = process_label(label)
            else:
                example["answer"] = process_label(-1)

            yield example["id"], example
