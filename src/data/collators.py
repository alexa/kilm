# Modified from Huggingface collators at
# https://github.com/huggingface/transformers/blob/v4.18.0/src/transformers/data/data_collator.py#L48

from dataclasses import dataclass
from typing import Dict, Optional, Union
import torch
from transformers import PreTrainedTokenizerBase, PreTrainedModel
from transformers.file_utils import PaddingStrategy


def trim_batch(
    input_ids,
    pad_token_id,
    attention_mask=None,
):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])


class CollatorWithTrimForApposGen:
    def __init__(self, pad_token_id, model, inference=False):
        """
        This class is for the zero-shot experiments on Appositive Generation task.
        The difference is that the original data collator will not pad "label_mask" 
        and "desc_labels".
        """
        self.pad_token_id = pad_token_id
        self.model = model
        self.inference = inference
    
    def __call__(self, batch) -> Dict[str, torch.Tensor]:
        input_ids = torch.stack([torch.LongTensor(x["input_ids"]) for x in batch])
        attention_mask = torch.stack([torch.LongTensor(x["attention_mask"]) for x in batch])
        labels = torch.stack([torch.LongTensor(x["labels"]) for x in batch])

        labels = trim_batch(labels, self.pad_token_id)
        input_ids, attention_mask = trim_batch(input_ids, self.pad_token_id, attention_mask=attention_mask)
        labels = self.ignore_pad_token_for_loss(labels, self.pad_token_id)

        # prepare decoder_input_ids
        if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=labels)

        if "label_mask" in batch[0]:
            label_masks = torch.stack([torch.LongTensor(x["label_mask"]) for x in batch])
            label_masks = label_masks[:, :labels.size(1)]
            labels[~label_masks.bool()] = -100
        
        if "desc_labels" in batch[0]:
            desc_labels = torch.stack([torch.LongTensor(x["desc_labels"]) for x in batch])
            desc_labels = trim_batch(desc_labels, self.pad_token_id)
            desc_label_masks = torch.stack([torch.LongTensor(x["desc_label_mask"]) for x in batch])
            desc_label_masks = desc_label_masks[:, :desc_labels.size(1)]
            desc_labels[~desc_label_masks.bool()] = -100


        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "decoder_input_ids":decoder_input_ids,
        }
        if self.inference:
            batch.update({"desc_labels": desc_labels, "label_mask": label_masks,})

        return batch
    
    def ignore_pad_token_for_loss(self, labels, pad_token_id):
        label_mask = labels.eq(pad_token_id)
        labels[label_mask.bool()] = -100
        return labels


@dataclass
class DataCollatorForSeq2Seq:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.
    Comparing with the original collator, we also pad "label_mask".
    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        model (:class:`~transformers.PreTrainedModel`):
            The model that is being trained. If set and has the `prepare_decoder_input_ids_from_labels`, use it to
            prepare the `decoder_input_ids`
            This is useful when using `label_smoothing` to avoid calculating loss twice.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[PreTrainedModel] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100

    def __call__(self, features):
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        label_masks = [feature["label_mask"] for feature in features] if "label_mask" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.

        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                feature["labels"] = (
                    feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                )

                if label_masks is not None:
                    mask_remainder = [0] * (max_label_length - len(feature["label_mask"]))
                    feature["label_mask"] = (
                        feature["label_mask"] + mask_remainder if padding_side == "right" else mask_remainder + feature["label_mask"]
                    )

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        input_ids, attention_mask = trim_batch(torch.LongTensor(features["input_ids"]), self.tokenizer.pad_token_id, attention_mask=torch.LongTensor(features["attention_mask"]))
        labels = trim_batch(torch.LongTensor(features["labels"]), -100)
        features["input_ids"] = input_ids
        features["attention_mask"] = attention_mask
        features["labels"] = labels
        if label_masks is not None:
            features["label_mask"] = torch.LongTensor(features["label_mask"])[:, :labels.size(1)] 

        # prepare decoder_input_ids
        if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        return features


@dataclass
class DataCollatorForSeq2SeqWithLeftPadding:
    """
    During the zero-shot evaluation with BART-style models, we need to provide the input 
    sequences with both encoder and decoder, which will make the positions where the 
    generation process starts in one batch, not same as each other. We conduct left-padding 
    to alliviate this issue with this data collator.
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[PreTrainedModel] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    trigger_length: int = 2

    def __call__(self, features):
        decoder_start_token_id = [self.model.config.decoder_start_token_id] if self.model is not None else []

        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        label_masks = [feature["label_mask"] for feature in features] if "label_mask" in features[0].keys() else None

        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.

        if labels is not None:
            assert label_masks is not None
            label_lengths = [l.index(1) if sum(l) != 0 else len(l) for l in label_masks]
            label_mask_lengths = [sum(l) for l in label_masks]
            max_label_length = max(label_lengths)
            max_label_mask_length = max(max(label_mask_lengths), 3) # TODO add here to avoid 0 length label mask
            for i, feature in enumerate(features):
                remainder = [self.tokenizer.pad_token_id] * (max_label_length - label_lengths[i])
                mask_remainder = [self.tokenizer.pad_token_id] * (max_label_mask_length - label_mask_lengths[i])
                padded_labels = remainder + feature["labels"]
                decoder_input_ids = remainder + decoder_start_token_id + feature["labels"]

                extra_length = max(0, max_label_length + self.trigger_length - len(padded_labels))
                split_index = max_label_length + self.trigger_length

                feature["labels"] = [self.tokenizer.pad_token_id] * extra_length + padded_labels[:split_index]
                feature["label_mask"] = \
                    padded_labels[split_index:split_index+label_mask_lengths[i]-self.trigger_length] + \
                    mask_remainder if len(padded_labels[split_index:split_index+label_mask_lengths[i]-self.trigger_length]) > 0 else mask_remainder[self.trigger_length:]
                feature["decoder_input_ids"] = \
                    [self.tokenizer.pad_token_id] * extra_length + \
                    decoder_input_ids[:max_label_length + self.trigger_length + len(decoder_start_token_id)]

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        input_ids, attention_mask = trim_batch(
            torch.LongTensor(features["input_ids"]),
            self.tokenizer.pad_token_id,
            attention_mask=torch.LongTensor(features["attention_mask"])
        )
        features["input_ids"] = input_ids
        features["attention_mask"] = attention_mask
        features["labels"] = torch.LongTensor(features["labels"])
        features["label_mask"] = torch.LongTensor(features["label_mask"])
        features["decoder_input_ids"] = torch.LongTensor(features["decoder_input_ids"])

        return features