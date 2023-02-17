from __future__ import absolute_import, division, print_function
import json
import os
import datasets
from .config import wiki_data_dir


_CITATION = """\
    N/A
"""

_DESCRIPTION = """\
    Wikipedia data
"""

_HOMEPAGE = "N/A"


_URLs = "N/A"


class Wikipedia(datasets.GeneratorBasedBuilder):

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="wiki_para",
            version=VERSION,
            description="Load wikipedia data (first several paragraphs)",
        ),
        datasets.BuilderConfig(
            name="wiki_doc",
            version=VERSION,
            description="Load wikipedia data for the entire document",
        ),
    ]

    DEFAULT_CONFIG_NAME = "wiki_para"

    def _info(self):
        features = datasets.Features(
            {
                "id": datasets.Value("string"),
                "title": datasets.Value("string"),
                "text": datasets.Value("string"),
                "desc": datasets.Value("string"),
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )
    
    def _split_generators(self, dl_manager):

        my_urls = _URLs

        # data_dir = dl_manager.download_and_extract(my_urls) 
        data_dir = wiki_data_dir # point to local dir to avoid downloading the dataset again
        if self.config.name == "wiki_para":
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={
                        "filepath": os.path.join(
                            data_dir, "first_paragraphs_filter_valid_unseen.json"
                        ),
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={
                        "filepath": os.path.join(
                            data_dir, "first_paragraphs_filter_valid_seen.json"
                        ),
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "filepath": os.path.join(
                            data_dir, "first_paragraphs_filter_train.json"
                        ),
                    },
                ),
            ]
        if self.config.name == "wiki_doc":
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={
                        "filepath": os.path.join(
                            data_dir, "paragraphs_filter_valid_unseen.json"
                        ),
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={
                        "filepath": os.path.join(
                            data_dir, "paragraphs_filter_valid_seen.json"
                        ),
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "filepath": os.path.join(
                            data_dir, "paragraphs_filter_train.json"
                        ),
                    },
                ),
            ]

    def _generate_examples(self, filepath):
        data = []
        with open(filepath, "r") as f:
            for line in f:
                data.append(json.loads(line))

        for idx, line in enumerate(data):

            sample = {
                "id": str(idx),
                "title": line["title"],
                "text": line["text"],
                "desc": line["desc"],
            }
            
            yield str(idx), sample
