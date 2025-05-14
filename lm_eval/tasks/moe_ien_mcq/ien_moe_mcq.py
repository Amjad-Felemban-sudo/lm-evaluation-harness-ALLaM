import csv
import os
import textwrap

import numpy as np
import pandas as pd 
import datasets
import sys
import ast
dirname = os.getcwd()
#print('dirname;;;;;;;;;;;;;;;;;;;;;;;;;',dirname)
DATA_ROOT = os.path.join(dirname, "lm_eval", "tasks", "moe_ien_mcq")
#print('DATA_ROOT;;;;;;;;;;;;;;;;;;;;;;;;;',DATA_ROOT)


_MOE_IEN_MCQs_CITATION = """\
@inproceedings{
    
}
"""

_MOE_IEN_MCQs_DESCRIPTION = """\


"""

DifficultyFactor = ['Easy_High_School', 'Easy_Primary_School', 'Easy_Intermediate_Level', 'Hard_High_School', 'Hard_Primary_School', 'Hard_Intermediate_Level', 'Intermediate_High_School', 'Intermediate_Primary_School', 'Intermediate_Intermediate_Level'] 

DifficultyFactor_ar = ['دون_المتوسط_الثانوية_مسارات', 'دون_المتوسط_المرحلة_الابتدائية', 'دون_المتوسط_المرحلة_المتوسطة', 'فوق_المتوسط_الثانوية_مسارات', 'فوق_المتوسط_المرحلة_الابتدائية', 'فوق_المتوسط_المرحلة_المتوسطة', 'متوسط_الثانوية_مسارات', 'متوسط_المرحلة_الابتدائية', 'متوسط_المرحلة_المتوسطة']

class moe_ien_mcq_Config(datasets.BuilderConfig):
     """BuilderConfig for moe_ien_mcq."""
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
        """BuilderConfig for moe_ien_mcq.
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
        super(moe_ien_mcq_Config, self).__init__(version=datasets.Version("1.0.0", ""), **kwargs)
        self.text_features = text_features
        self.label_column = label_column
        self.label_classes = label_classes
        self.data_url = data_url
        self.data_dir = data_dir
        self.citation = citation
        self.url = url
        self.process_label = process_label


# TODO: Name of the dataset usually matches the script name with CamelCase instead of snake_case
class moe_ien_mcq(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [

        moe_ien_mcq_Config(
            name="moe_ien_mcq", 
            description="Dataset selected from MOE platform IEN, they are arranged by grade/topic/difficulty level. They cover all the Saudi curriculum from 1st grade to high school. The dataset contains 10435 questions in MCQ format ",

                text_features={"Question": "Question",
                               "Choices": "Choices", 
                               "Subject":"Subject",
                               "Stage":"Stage",
                               "Grade":"Grade",
                               "DifficultyFactor": "DifficultyFactor",
                               "domain": "domain",
                              }, 
                label_classes=["1", "2", "3", "4", "5", "6"],
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
        ),
    ]


    def _info(self):
        features = {"Question": datasets.Value("string"),
                   "Choices":datasets.features.Sequence(datasets.Value("string"))}
        features["Answer"] = datasets.Value("string")
        features["idx"] = datasets.Value("int32")
        features["Subject"] = datasets.Value("string")
        features["Stage"] = datasets.Value("string")
        features["Grade"] = datasets.Value("string")
        features["DifficultyFactor"] = datasets.Value("string")
        features["domain"] = datasets.Value("string")
        return datasets.DatasetInfo(
                description=_MOE_IEN_MCQs_DESCRIPTION,
                features=datasets.Features(features),
                homepage=self.config.url,
                citation=self.config.citation + "\n" +_MOE_IEN_MCQs_CITATION,
        )

    def _split_generators(self, dl_manager):
        list_splits = []
        list_splits.append(
            datasets.SplitGenerator(
            name=datasets.Split.TEST,
            gen_kwargs={
                "data_file": os.path.join(DATA_ROOT,"mcq_test.xlsx"),
                "split": "test",
            },
            )
        )
        list_splits.append(
            datasets.SplitGenerator(
            name=datasets.Split.VALIDATION,
            gen_kwargs={
                "data_file": os.path.join(DATA_ROOT,"mcq_dev.xlsx"),
                "split": "dev",
            },
            )
        )
        return list_splits

    # def _generate_examples(self, data_file, split, mrpc_files=None):
    #     process_label = self.config.process_label
    #     label_classes = self.config.label_classes
    #     df = pd.read_excel(data_file)
    #     i = 0
    #     for n, row in df.iterrows():
    #         choices_list = [row['choice_1'],row['choice_2'], row['choice_3'],row['choice_4'], row['choice_5'], row['choice_6']]
    #         row = {
    #                 "Question": row['Question'],
    #                 "Choices": [item for item in choices_list if item != "-"],
    #                 "Answer": ast.literal_eval(row['Answer'])[0],
    #                 "Subject": row['Speciality'],
    #                 "Stage": row['Stage'],
    #                 "Grade": row['Level'],
    #                 "DifficultyFactor":row['DifficultyFactor'],
    #                 "domain": f"{row['Stage']}_{row['Speciality']}",
    #             }
                
    #         example = {feat: row[col] for feat, col in self.config.text_features.items()}
    #         example["idx"] = i
    #         i+=1
    #         if self.config.label_column in row:
    #             label = row[self.config.label_column]
    #             if label_classes and label in label_classes:
    #                 label = int(label) if label else None
    #                 example["Answer"] = process_label(label)
    #             else:
    #                 example["Answer"] = process_label(-1)
                
    #             # Filter out corrupted rows.
    #             for value in example.values():
    #                 if value is None:
    #                     break
    #             else:
    #                 yield example["idx"],example


    def _generate_examples(self, data_file, split):
        df = pd.read_excel(data_file)
        df = df.fillna("")  # so you don't get float('nan') anywhere

        idx_counter = 0
        for row_idx, row in df.iterrows():
            question = str(row["Question"])

            choices_list = [
                str(row["choice_1"]), 
                str(row["choice_2"]), 
                str(row["choice_3"]), 
                str(row["choice_4"]), 
                str(row["choice_5"]), 
                str(row["choice_6"]),
            ]
            choices_list = [c for c in choices_list if c != "-" and c.strip() != ""]

            if not question.strip() or not choices_list:
                continue

            subject = str(row["Speciality"])
            stage = str(row["Stage"])
            grade = str(row["Level"])
            difficulty = str(row["DifficultyFactor"])

            # Now parse the answer (if stored as JSON-like text)
            # or skip the row if it's missing
            try:
                raw_answers = ast.literal_eval(row["Answer"])
                raw_answer = raw_answers[0] if raw_answers else ""
            except Exception:
                continue  # skip if there's a parse error

            raw_answer = str(raw_answer)

            example = {
                "Question": question,
                "Choices": choices_list,
                "Subject": subject,
                "Stage": stage,
                "Grade": grade,
                "DifficultyFactor": difficulty,
                "domain": f"{stage}_{subject}",
                "idx": idx_counter,
            }
            idx_counter += 1

            label_classes = self.config.label_classes
            if raw_answer in label_classes:
                example["Answer"] = raw_answer
            else:
                continue

            yield example["idx"], example


