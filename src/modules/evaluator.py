# This code is mostly modified from:
# https://github.com/huggingface/datasets/blob/master/metrics/squad/evaluate.py

import os
import logging
import re
import string
from collections import Counter

from src.utils.utils import (
    save_json,
)

try:
    import nltk
    NLTK_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    NLTK_AVAILABLE = False


logger = logging.getLogger(__name__)


def normalize_answer(s):
    """
    Post-process the generations to compute the overlap between generations and the references.
    Lower text and remove punctuation, articles and extra whitespace.
    """

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


"""Calculate the best score over the different references"""
def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate(predictions, references):
    f1 = exact_match = total = 0
    for prediction, ground_truths in zip(predictions, references):
        exact_match += metric_max_over_ground_truths(exact_match_score, prediction, ground_truths)
        f1 += metric_max_over_ground_truths(f1_score, prediction, ground_truths)
        total += 1

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {"exact_match": exact_match, "f1": f1}


class F1Evaluator():
    def __init__(self) -> None:
        """ 
        This class is written because there's no easy function in HuggingFace datasets package 
        to evaluation uni-gram F1.
        """
        pass

    def compute(self, predictions, references):
        """Calculate the Exact Match and F1 scores."""
        metric = evaluate(predictions, references)
        return metric


# Save results into files
def handle_metrics(split, metrics, output_dir):

    """
    Log and save metrics
    Args:
    - split: one of train, val, test
    - metrics: metrics dict
    - output_dir: where to save the metrics
    """

    logger.info(f"***** {split} metrics *****")
    for key in sorted(metrics.keys()):
        logger.info(f"  {key} = {metrics[key]}")
    save_json(metrics, os.path.join(output_dir, f"{split}_results.json"))
