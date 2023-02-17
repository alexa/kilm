import re
import os
import math
import random
import contextlib
import numpy as np
from collections import defaultdict
import torch
from transformers.utils import logging

logger = logging.get_logger(__name__)


@contextlib.contextmanager
def numpy_seed(seed, *addl_seeds):
    """
    Context manager which seeds the NumPy PRNG with the specified seed and
    restores the state afterward
    This function is from fairseq
    """
    if seed is None:
        yield
        return
    if len(addl_seeds) > 0:
        seed = int(hash((seed, *addl_seeds)) % 1e6)
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def split_line(examples, tokenizer, max_source_length=512):
    """
    Split long articles in to sub-articles within `max_source_length`;
    Map list of list of features after spliting into list of features.
    """
    tokenized_examples = tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_source_length,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        return_special_tokens_mask=False,
        return_token_type_ids=False,
        padding=False,
    )
    decode_text = tokenizer.batch_decode(tokenized_examples["input_ids"], skip_special_tokens=True,
                                         clean_up_tokenization_spaces=False)

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    new_samples = {
        "id": [],
        "title": [],
        "text": [],
        "desc": [],
        "sample_idx": [],
    }
    for i in range(len(tokenized_examples["input_ids"])):
        example_id = sample_mapping[i]

        new_samples["id"].append(examples["id"][example_id])
        new_samples["title"].append(examples["title"][example_id])
        new_samples["desc"].append(examples["desc"][example_id])
        new_samples["sample_idx"].append(i)

        new_samples["text"].append(decode_text[i])

    return new_samples


def proc_wiki_lm_dataset(
        training_args,
        data_args,
        datasets,
        dataset_dir,
        tokenizer,
        epochs=1,
        entity_bos_token="<ent>",
        entity_eos_token="</ent>",
        desc_bos_token="<ent_desc>",
        desc_eos_token="</ent_desc>",
        mlm_probability=0.15,
        poisson_lambda=3.0,
        generate=False,
        **kwargs):

    """
    The tokenization function for preparing the data samples for pre-training
    """
    if training_args.do_train:
        column_names = datasets["train"].column_names
    else:
        column_names = datasets["validation"].column_names

    pad_on_right = tokenizer.padding_side == "right"

    def prepare_split_samples(examples):
        splited_samples = split_line(examples, tokenizer, max_source_length=data_args.max_source_length)
        return splited_samples

    def copy_data_by_epochs(examples):
        # Copy the data into epoch times in advance for dynamic masking.
        extended_examples = defaultdict(list)
        for _ in range(epochs):
            for key, value in examples.items():
                extended_examples[key].extend(value.copy())
        return dict(extended_examples)

    def convert_desc_format(desc_line):
        # Convert the format of the short descriptions into a hash table.
        desc_dict = {}
        entities = desc_line.split("<sep_ent>")
        for entity in entities:
            desc_dict[entity.split("<sep>")[0]] = entity.split("<sep>")[1] + "<sep>" + entity.split("<sep>")[
                2]  # TODO add wikipedia title and sep token
        return desc_dict

    def _replace_all_leave_one(src, tgt, entities, entity_index, desc_dict):
        # Replace the selected entity(ies), while the rest of the entities are converted into their surface forms.
        item, ent, content, span = entities[entity_index]
        if item not in desc_dict:
            src = src.replace(item, content)
            tgt = tgt.replace(item, content)
        else:
            pref = entity_bos_token + content + entity_eos_token + desc_bos_token
            appe = desc_eos_token
            content_len = len(tokenizer.encode(desc_dict[item], add_special_tokens=False))

            prefix = src[:span[0]]
            appendix = src[span[1]:]
            src = prefix + pref + "<mask>" * content_len + appe + appendix
            tgt = prefix + pref + desc_dict[item] + appe + appendix

        for idx, (item, ent, content, _) in enumerate(entities):
            if idx != entity_index:
                src = src.replace(item, content)
                tgt = tgt.replace(item, content)
        return src, tgt

    def _entity_masking(texts, descs, copy=False):
        # Sample the entities in the paragraphs and process the paragraph into a special structure.
        srcs = []
        tgts = []
        for sent, desc_line in zip(texts, descs):
            src, tgt = sent, sent
            entities = detect_entity(sent)
            if entities is not None:
                # only keep the part with entities
                desc_dict = convert_desc_format(desc_line)
                if copy:
                    for entity_index in range(min(len(entities), 10)):
                        _src, _tgt = _replace_all_leave_one(src, tgt, entities=entities, entity_index=entity_index,
                                                            desc_dict=desc_dict)
                        srcs.append(tokenizer.bos_token + _src + tokenizer.eos_token)
                        tgts.append(tokenizer.bos_token + _tgt + tokenizer.eos_token)
                else:
                    entity_index = random.sample(range(len(entities)), 1)[0]
                    src, tgt = _replace_all_leave_one(src, tgt, entities=entities, entity_index=entity_index,
                                                      desc_dict=desc_dict)
                    srcs.append(tokenizer.bos_token + src + tokenizer.eos_token)
                    tgts.append(tokenizer.bos_token + tgt + tokenizer.eos_token)
            else:
                srcs.append(tokenizer.bos_token + src + tokenizer.eos_token)
                tgts.append(tokenizer.bos_token + tgt + tokenizer.eos_token)

        return srcs, tgts

    # This function is mostly adopted from
    # https://github.com/morganmcg1/rotobart/blob/main/data_collator.py#L67
    def tokenize(examples, copy=False):
        texts = examples["text"]
        descs = examples["desc"]

        srcs, tgts = _entity_masking(texts, descs, copy=copy)

        tokenized_examples = tokenizer(
            srcs,
            add_special_tokens=False,
            truncation=True,
            max_length=data_args.max_source_length,
            padding=True,
        )
        tokenized_tgts = tokenizer(
            tgts,
            add_special_tokens=False,
            truncation=True,
            max_length=data_args.max_source_length,
            padding=True if not generate else False,
        )

        special_tokens_mask = [
            tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in
            tokenized_examples["input_ids"]
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

        inputs = torch.tensor(tokenized_examples["input_ids"])
        attention_mask = torch.tensor(tokenized_examples["attention_mask"])
        if not generate:
            labels = torch.tensor(tokenized_tgts["input_ids"])

        # determine how many tokens we need to mask in total
        is_token = ~(inputs == tokenizer.pad_token_id) & ~special_tokens_mask
        num_to_mask = int(math.ceil(is_token.float().sum() * mlm_probability))

        if num_to_mask == 0:
            # extract mask tokens
            mask = inputs == tokenizer.mask_token_id
            # remove mask tokens that are not starts of spans
            to_remove = mask.bool() & mask.bool().roll(1, 1)
            new_inputs = torch.full_like(inputs, fill_value=tokenizer.pad_token_id)
            for i, example in enumerate(torch.split(inputs, split_size_or_sections=1, dim=0)):
                new_example = example[0][~to_remove[i]]
                new_inputs[i, 0:new_example.shape[0]] = new_example

            new_attention_mask = torch.full_like(attention_mask, fill_value=0)
            for i, example in enumerate(torch.split(attention_mask, split_size_or_sections=1, dim=0)):
                new_example = example[0][~to_remove[i]]
                new_attention_mask[i, 0:new_example.shape[0]] = new_example

            tokenized_examples["input_ids"] = new_inputs.tolist()
            tokenized_examples["attention_mask"] = new_attention_mask.tolist()

            mask = mask | (inputs == tokenizer.mask_token_id) | (
                        inputs == tokenizer.convert_tokens_to_ids(desc_bos_token)) | (
                               inputs == tokenizer.convert_tokens_to_ids(desc_eos_token))
            if generate:
                label_lens = [len(l) for l in tokenized_tgts["input_ids"]]
                label_mask = [m[:label_lens[idx]] for idx, m in enumerate(mask.tolist())]
                tokenized_examples["labels"] = tokenized_tgts["input_ids"]
                tokenized_examples["label_mask"] = label_mask
            else:
                label_mask = torch.full_like(labels, fill_value=0)
                label_mask[:mask.shape[0], :mask.shape[1]] = mask

                label_pad_mask = labels.eq(tokenizer.pad_token_id)
                labels[label_pad_mask.bool()] = -100

                tokenized_examples["labels"] = labels.tolist()
                tokenized_examples["label_mask"] = label_mask.tolist()
            return tokenized_examples

        # generate a sufficient number of span lengths
        poisson_distribution = torch.distributions.Poisson(rate=poisson_lambda)
        lengths = poisson_distribution.sample(sample_shape=(num_to_mask,))
        while torch.cumsum(lengths, 0)[-1] < num_to_mask:
            lengths = torch.cat([lengths, poisson_distribution.sample(sample_shape=(num_to_mask,))])

        # remove all spans of length 0
        # Note that BART inserts additional mask tokens where length == 0,
        # which we do not implement for now as it adds additional complexity
        lengths = lengths[lengths > 0]

        # trim to about num_to_mask tokens
        idx = torch.argmin(torch.abs(torch.cumsum(lengths, 0) - num_to_mask)) + 1
        lengths = lengths[:idx + 1]

        # select span start indices
        token_indices = is_token.nonzero(as_tuple=False)
        span_starts = torch.randperm(token_indices.shape[0])[:lengths.shape[0]]

        # prepare mask
        masked_indices = token_indices[span_starts]
        mask = torch.full_like(inputs, fill_value=False)

        # mask span start indices
        for mi in masked_indices:
            mask[tuple(mi)] = True
        lengths -= 1

        # fill up spans
        max_index = inputs.shape[1] - 1
        remaining = (lengths > 0) & (masked_indices[:, 1] < max_index)
        while torch.any(remaining):
            masked_indices[remaining, 1] += 1
            for mi in masked_indices:
                mask[tuple(mi)] = True
            lengths -= 1
            remaining = (lengths > 0) & (masked_indices[:, 1] < max_index)

        # place the mask tokens
        mask[special_tokens_mask] = False
        inputs[mask.bool()] = tokenizer.mask_token_id

        # add the pre-added mask tokens into mask
        mask = mask | (inputs == tokenizer.mask_token_id)

        # remove mask tokens that are not starts of spans
        to_remove = mask.bool() & mask.bool().roll(1, 1)
        new_inputs = torch.full_like(inputs, fill_value=tokenizer.pad_token_id)
        for i, example in enumerate(torch.split(inputs, split_size_or_sections=1, dim=0)):
            new_example = example[0][~to_remove[i]]
            new_inputs[i, 0:new_example.shape[0]] = new_example

        new_attention_mask = torch.full_like(attention_mask, fill_value=0)
        for i, example in enumerate(torch.split(attention_mask, split_size_or_sections=1, dim=0)):
            new_example = example[0][~to_remove[i]]
            new_attention_mask[i, 0:new_example.shape[0]] = new_example

        mask = mask | (inputs == tokenizer.convert_tokens_to_ids(desc_bos_token)) | (
                    inputs == tokenizer.convert_tokens_to_ids(desc_eos_token))
        if not generate:
            label_mask = torch.full_like(labels, fill_value=0)
            label_mask[:mask.shape[0], :mask.shape[1]] = mask

            label_pad_mask = labels.eq(tokenizer.pad_token_id)
            labels[label_pad_mask.bool()] = -100
        else:
            label_lens = [len(l) for l in tokenized_tgts["input_ids"]]
            label_mask = [m[:label_lens[idx]] for idx, m in enumerate(mask.tolist())]

        tokenized_examples["input_ids"] = new_inputs.tolist()
        tokenized_examples["attention_mask"] = new_attention_mask.tolist()
        tokenized_examples["labels"] = labels.tolist() if not generate else tokenized_tgts["input_ids"]
        tokenized_examples["label_mask"] = label_mask.tolist() if not generate else label_mask

        assert len(tokenized_examples["input_ids"]) == len(tokenized_examples["attention_mask"]) == len(
            tokenized_examples["labels"]) == len(tokenized_examples["label_mask"])
        return tokenized_examples

    def prepare_train_features(examples):
        """Prepare the features for training"""
        splited_samples = prepare_split_samples(examples)
        extended_samples = copy_data_by_epochs(splited_samples)
        tokenized_examples = tokenize(extended_samples)
        return tokenized_examples

    def prepare_valid_features(examples):
        """Prepare the features for evaluation"""
        splited_samples = prepare_split_samples(examples)
        tokenized_examples = tokenize(splited_samples, copy=True)
        return tokenized_examples

    if training_args.do_train:
        print("Processing training dataset!")
        train_dataset = datasets["train"].map(
            prepare_train_features,
            batched=True,
            batch_size=100,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )
        os.makedirs(dataset_dir + '/train', exist_ok=True)
        logger.info(f"Saving dataset {dataset_dir + '/train'} with {len(train_dataset)} samples to disk.")
        train_dataset.save_to_disk(dataset_dir + '/train')

    else:
        train_dataset = None

    if training_args.do_eval:
        print("Processing validation dataset!")
        validation_dataset = datasets["validation"].map(
            prepare_valid_features,
            batched=True,
            batch_size=100,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )
        os.makedirs(dataset_dir + '/valid', exist_ok=True)
        logger.info(f"Saving dataset {dataset_dir + '/valid'} with {len(validation_dataset)} samples to disk.")
        validation_dataset.save_to_disk(dataset_dir + '/valid')

    else:
        validation_dataset = None

    if training_args.do_predict:
        print("Spliting test dataset!")
        test_dataset = datasets["test"].map(
            prepare_valid_features,
            batched=True,
            batch_size=100,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )
        os.makedirs(dataset_dir + '/test', exist_ok=True)
        logger.info(f"Saving dataset {dataset_dir + '/test'} with {len(test_dataset)} samples to disk.")
        test_dataset.save_to_disk(dataset_dir + '/test')

    else:
        test_dataset = None

    return train_dataset, validation_dataset, test_dataset


def detect_entity(text):
    """
    match with pattern ``[[.*?]]'' to collect all the entities
    """

    entities = []
    spans = []
    for e in re.finditer(r"\[\[.*?\]\]", text):
        entities.append(e.group())
        spans.append((e.start(), e.end()))

    ents = []
    for entity, sp in zip(entities, spans):
        temp = entity[2:-2].split("|")
        # assert len(temp) == 2 or len(temp) == 1, ValueError(f"Unexpected Entity: {entity}")
        if len(temp) == 2 or len(temp) == 1:
            ent, content = temp[0], temp[-1]
            ent = ent.split("#")[0] if len(ent.split("#")[0]) != 0 else ent
            ent = ent[0].upper() + ent[1:]
            ents.append((entity, ent, content, sp))
    if len(ents) == 0:
        return None

    return ents


proc_wiki_dataset = {
    "lm": proc_wiki_lm_dataset,
    "task": None,
}
