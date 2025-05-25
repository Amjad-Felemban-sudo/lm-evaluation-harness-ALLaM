# coding=utf-8
# https://github.com/huggingface/datasets/blob/main/templates/new_dataset_script.py
# Lint as: python3
"""The Arabic mmlu from aceGPT benchmark."""


import csv
import os
import textwrap

import pandas as pd
import numpy as np

import datasets
from pathlib import Path

dirname = os.getcwd()
DATA_ROOT = os.path.join(dirname, "lm_eval", "tasks", "etec")

_ETEC_CITATION = """\
@inproceedings{
    
}
"""

_ETEC_DESCRIPTION = """\


"""


class ETECConfig(datasets.BuilderConfig):
    """BuilderConfig for ETEC."""

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
        """BuilderConfig for ETECConfig.

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
        super(ETECConfig, self).__init__(version=datasets.Version("1.0.0", ""), **kwargs)
        self.text_features = text_features
        self.label_column = label_column
        self.label_classes = label_classes
        self.data_url = data_url
        self.data_dir = data_dir
        self.citation = citation
        self.url = url
        self.process_label = process_label


class ETEC(datasets.GeneratorBasedBuilder):
    """The ETEC benchmark."""

    BUILDER_CONFIGS = [
        
        ETECConfig(
                name= "etec",
                description=textwrap.dedent("etec."),
                text_features={"id":"id","question":"question","choices":"choices"}, 
                label_classes=["أ", "ب", "ج", "د"],
                label_column="label",
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
                     "choices":datasets.features.Sequence(datasets.Value("string"))

                   }
        features["label"] = datasets.Value("string")
        return datasets.DatasetInfo(
            description=_ETEC_DESCRIPTION,
            features=datasets.Features(features),
            homepage=self.config.url,
            citation=self.config.citation + "\n" + _ETEC_CITATION,
        )

    def _split_generators(self, dl_manager):
        list_splits = []
        
        list_splits.append(
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={
                        "data_file": os.path.join(DATA_ROOT , "test.xlsx"),
                        "split": 'test',
                    },
                )
            )

          
        return list_splits
    
    def _generate_examples(self, data_file, split, mrpc_files=None):
        process_label = self.config.process_label
        label_classes = self.config.label_classes
        
        print(data_file)
        reader = pd.read_excel(data_file)
        for n, row in reader.iterrows():
            row = {
                "id":n,
                    "question": row["questions"].strip(),
                    "choices": [str(row['Option1']).strip(),str(row['Option2']).strip(),str(row['Option3']).strip(),str(row['Option4']).strip()],
                    "label": row["answer"].strip()
                }
            
            example = {feat: row[col] for feat, col in self.config.text_features.items()}
            if self.config.label_column in row:
                label = row[self.config.label_column]
                # For some tasks, the label is represented as 0 and 1 in the tsv
                # files and needs to be cast to integer to work with the feature.
                if label_classes and label not in label_classes:
                    label = int(label) if label else None
                example["label"] = process_label(label)
            else:
                print(row)
                example["label"] = process_label(-1)

            # Filter out corrupted rows.
            for value in example.values():
                if value is None:
                    break
            else:
                yield example["id"], example
