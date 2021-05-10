# Lint as: python3
"""ARSC few-shot sentiment classification dataset."""

from __future__ import absolute_import, division, print_function

import csv

import datasets


_DESCRIPTION = """\
The diverse few-shot learning dataset constructed from Amazon Review.
Data for the NAACL 2018 paper: Diverse Few-Shot Text Classification with Multiple Metrics
The four domains ``books, dvd, electronics, kitchen_housewares'' are heldout as testing few-shot tasks. Note that the dev sets of these tasks are not used during meta-testing.
"""

_CITATION = """\
@article{yu2018diverse,
  title={Diverse few-shot text classification with multiple metrics},
  author={Yu, Mo and Guo, Xiaoxiao and Yi, Jinfeng and Chang, Shiyu and Potdar, Saloni and Cheng, Yu and Tesauro, Gerald and Wang, Haoyu and Zhou, Bowen},
  journal={arXiv preprint arXiv:1805.07513},
  year={2018}
}
"""

_DATA_DOWNLOAD_URL = "https://github.com/Gorov/DiverseFewShot_Amazon"


class ARSC(datasets.GeneratorBasedBuilder):
    """ARSC few-shot sentiment classification dataset."""

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "text": datasets.Value("string"),
                    "label": datasets.features.ClassLabel(names=["1", "-1"]),
                }
            ),
            homepage=_DATA_DOWNLOAD_URL,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        train_path = dl_manager.download_and_extract(_TRAIN_DOWNLOAD_URL)
        test_path = dl_manager.download_and_extract(_TEST_DOWNLOAD_URL)
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": train_path}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": test_path}),
        ]

    def _generate_examples(self, filepath):
        """Generate AG News examples."""
        with open(filepath, encoding="utf-8") as csv_file:
            csv_reader = csv.reader(
                csv_file, quotechar='"', delimiter=",", quoting=csv.QUOTE_ALL, skipinitialspace=True
            )
            for id_, row in enumerate(csv_reader):
                label, title, description = row
                # Original labels are [1, 2, 3, 4] ->
                #                   ['World', 'Sports', 'Business', 'Sci/Tech']
                # Re-map to [0, 1, 2, 3].
                label = int(label) - 1
                text = " ".join((title, description))
                yield id_, {"text": text, "label": label}