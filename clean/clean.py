"""Conceptual texts dataset."""
import os
import csv
import textwrap

import datasets


_DESCRIPTION = """\
Clean images + sanitized prompts
"""

class CleanDataset(datasets.GeneratorBasedBuilder):
    """Builder for image-text dataset for fine-tuning SD."""

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = []
    for clean_d in ['dog', 'cat','cartoon_dog', 'cartoon_automobile']:
            for data_type in ['keyword']: 
                BUILDER_CONFIGS.append(datasets.BuilderConfig(f"{data_type}_{clean_d}", version=VERSION))

    DEFAULT_CONFIG_NAME = "keyword_cat"

    def _info(self):
        features = datasets.Features(
            {
                "image": datasets.Image(),
                "text": datasets.Value("string"),
            },
        )   
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        annotations_file = f"data/dataset_csv/clean_{self.config.name}_train_set.csv"
        splits = [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
            "annotations_file": annotations_file,
            "image_dir": ".",
        },
            ),
        ]
        return splits

    def _generate_examples(self, annotations_file, image_dir):
        
        with open(annotations_file, encoding="utf-8") as f:
            for i, row in enumerate(csv.reader(f, delimiter=",")):
                assert len(row) == 2
                image_path, text = row
                yield i, {
                    "image": os.path.join(image_dir, image_path),
                    "text": text,
                },
                
                
                
        

