# coding=utf-8
# https://github.com/huggingface/datasets/blob/main/templates/new_dataset_script.py
# Lint as: python3
"""The araTruthfulQA benchmark."""

import csv
import os
import textwrap
import pandas as pd
import numpy as np

import datasets
from pathlib import Path
dirname = os.getcwd()

DATA_ROOT = os.path.join(dirname, "lm_eval", "tasks", "araTruthfulQA")

_araTruthfulQA_CITATION = """\
@inproceedings{
    
}
"""

_araTruthfulQA_DESCRIPTION = """\


"""


class araTruthfulQAConfig(datasets.BuilderConfig):
    """BuilderConfig for araTruthfulQA."""

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
        """BuilderConfig for araTruthfulQAConfig.

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

        super(araTruthfulQAConfig, self).__init__(version=datasets.Version("1.0.0", ""), **kwargs)
        self.text_features = text_features
        self.label_column = label_column
        self.label_classes = label_classes
        self.data_url = data_url
        self.data_dir = data_dir
        self.citation = citation
        self.url = url
        self.process_label = process_label


class araTruthfulQA(datasets.GeneratorBasedBuilder):
    """The araTruthfulQA benchmark."""

    BUILDER_CONFIGS = [
        araTruthfulQAConfig(
            name="araTruthfulQA",
            description=textwrap.dedent("araTruthfulQA."),
            text_features={"id": "id", "input": "input", "options": "options", "answer":"answer", "label": "label"},
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
        features = {
            "id": datasets.Value("int32"),
            "input": datasets.Value("string"),
            "options": datasets.features.Sequence(datasets.Value("string")),
            "answer": datasets.Value("string"),
            "label": datasets.Value("string"),
        }
        return datasets.DatasetInfo(
            description=_araTruthfulQA_DESCRIPTION,
            features=datasets.Features(features),
            homepage=self.config.url,
            citation=self.config.citation + "\n" + _araTruthfulQA_CITATION,
        )

    def _split_generators(self, dl_manager):
        list_splits = []

        list_splits.append(
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "data_file": os.path.join(DATA_ROOT, "AraTruthfulQA-both_test.jsonl"),
                    "split": 'test',
                },
            )
        )
        list_splits.append(
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "data_file": os.path.join(DATA_ROOT, "AraTruthfulQA-both_dev.jsonl"),
                    "split": 'dev',
                },
            )
        )
        return list_splits

    def _generate_examples(self, data_file, split, mrpc_files=None):
        print(data_file)
        reader = pd.read_json(data_file, lines=True)
        print(self.config.text_features.items())
        for n, row in reader.iterrows():
            row = {
                "id": row["iid"],
                "input": row["input"].strip(),
                "options": row["options"],
                "answer": row["answer"],
                "label": row["label"],
            }
            example = {feat: row[col] for feat, col in self.config.text_features.items()}

            for value in example.values():
                if value is None:
                    break
            else:
                yield n, example