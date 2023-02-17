import os 
import json
import numpy as np
from tqdm import tqdm

import torch


class ApposProcessor():
    def __init__(self, data_path, cache_dir) -> None:
        """
        The data processor for Appostive Generation (data loading & tokenization)
        """
        self.path = data_path
        self.cache_dir = cache_dir
        self.trim_end = True

    @staticmethod
    def load_data_from_file(data_file):
        with open(data_file, "r") as f:
            data = []
            for line in f.readlines():
                data.append(json.loads(line))
        return data

    @staticmethod
    def tokenize_data(tokenizer, inputs, label, desc_label, max_length=1024):
        tokenized_input = tokenizer.encode(inputs)
        attention_mask = [1] * len(tokenized_input)

        mask_index = tokenized_input.index(tokenizer.convert_tokens_to_ids("<mask>")) + 1
        max_fore_len = min(mask_index, int(max_length)/2)
        truncate_length = int(mask_index - max_fore_len)

        tokenized_label = tokenizer.encode(label)
        label_eos_index = tokenized_label.index(tokenizer.eos_token_id)
        labek_mask = \
            [0] * mask_index + \
            [1] * (label_eos_index - mask_index + 1) + \
            [0] * (len(tokenized_label) - label_eos_index - 1)

        tokenized_desc_label = tokenizer.encode(desc_label)
        desc_label_eos_index = tokenized_desc_label.index(tokenizer.eos_token_id)
        desc_label_mask = \
            [0] * mask_index + \
            [1] * (desc_label_eos_index - mask_index + 1) + \
            [0] * (len(tokenized_desc_label) - desc_label_eos_index - 1)
        
        if truncate_length > 0:
            tokenized_input, attention_mask = tokenized_input[truncate_length:], attention_mask[truncate_length:]
            tokenized_label, labek_mask, tokenized_desc_label, desc_label_mask = \
                tokenized_label[truncate_length:], labek_mask[truncate_length:], \
                tokenized_desc_label[truncate_length:], desc_label_mask[truncate_length:]

        assert len(labek_mask) == len(tokenized_label)
        assert len(desc_label_mask) == len(tokenized_desc_label)

        return tokenized_input, attention_mask, tokenized_label, labek_mask, tokenized_desc_label, desc_label_mask

    @staticmethod
    def get_max_length(samples):
        max_length = 0
        for sample in samples:
            for k in ["input_ids", "labels"]:
                max_length = max(len(sample[k]), max_length)
        return max_length

    def pad_sequences(self, tokenizer, samples, max_length):
        padded_samples = []
        for sample in samples:
            padded_sample = {}
            for k, v in sample.items():
                remaining = max_length - len(v)
                pad_token_id = tokenizer.pad_token_id if k in ["input_ids", "labels", "desc_labels"] else 0

                if self.trim_end:
                    v = v[:max_length] if remaining <= 0 else v + [pad_token_id] * remaining
                else:
                    v = v[-max_length:] if remaining <= 0 else v + [pad_token_id] * remaining

                padded_sample[k] = v
            padded_samples.append(padded_sample)
        return padded_samples

    def preprocess_data(
            self,
            data,
            tokenizer,
            add_ent_tokens=False,
            add_desc_tokens=False,
            add_fact=False,
            add_next=True,
            max_length=512,
            ent_token="<ent>"
    ):
        """
        Pre-process and tokenize the data samples based on different settings
        """
        samples = []
        for sample in tqdm(data, total=len(data)):
            prev_sent = "" if len(sample["previous_sentence"]) == 0 else " ".join(sample["previous_sentence"][0][-100:])
            next_sent = "" if len(sample["next_sentence"]) == 0 else " ".join(sample["next_sentence"][:100])
            curr_sent = " ".join(sample["input"])

            name = sample["name"]

            assert "<appos>" in curr_sent

            # match input
            if add_ent_tokens and len(name.strip()) > 0:
                # This condition is just for the case when we use KILM model.
                sub_curr_sent = curr_sent[:curr_sent.index("<appos>")]
                index = sub_curr_sent.rfind(name)
                new_name = ent_token + name + ent_token[0] + "/" + ent_token[1:]
                curr_sent = curr_sent[:index] + new_name + curr_sent[index+len(name):]

            if " <appos>" in curr_sent:
                if add_desc_tokens:
                    curr_sent = curr_sent.replace(" <appos>", "<ent_desc><mask></ent_desc>")
                else:
                    curr_sent = curr_sent.replace(" <appos>", " <mask>")
            else:
                if add_desc_tokens:
                    curr_sent = curr_sent.replace("<appos>", "<ent_desc><mask></ent_desc>")
                else:
                    curr_sent = curr_sent.replace("<appos>", " <mask>")

            input_seq = prev_sent + " " + curr_sent + " " + next_sent if add_next else prev_sent + " " + curr_sent

            target = "<EMPTY>" if len(sample["target"]) == 0 else " ".join(sample["target"])
            description = sample["descriptions"] if len(sample["descriptions"]) > 0 else ""

            if "used" in sample and True in sample["used"]:
                index = sample["used"].index(True)
                property = sample["properties"][index]
                value = sample["values"][index]
                qualifier = " ".join([" ".join(v) for v in sample["qualifiers"][index]])
            else:
                property = ""
                value = ""
                qualifier = ""
            
            if add_fact:
                fact = " ".join([" <sep>", property, value])
                if len(qualifier) > 0:
                    fact += " <sep> " + qualifier

                input_seq += fact
            
            label = input_seq.replace("<mask>", target+tokenizer.eos_token)
            desc_label = input_seq.replace("<mask>", description+tokenizer.eos_token)
            
            tokenized_input, attention_mask, tokenized_label, labek_mask, tokenized_desc_label, desc_label_mask = \
                self.tokenize_data(tokenizer, input_seq, label,
                                                                                                                                     desc_label, max_length=max_length)

            new_sample = {
                "input_ids": tokenized_input,
                "attention_mask": attention_mask,
                "labels": tokenized_label,
                "label_mask": labek_mask,
                "desc_labels": tokenized_desc_label,
                "desc_label_mask": desc_label_mask,
            }
            samples.append(new_sample)

        max_seq_length = self.get_max_length(samples)
        max_seq_length = min(max_seq_length, max_length)
        padded_samples = self.pad_sequences(tokenizer, samples, max_seq_length)
        
        return padded_samples

    def load_dataset(
            self,
            dataset_config,
            tokenizer,
            add_ent_tokens=False,
            add_desc_tokens=False,
            add_fact=False,
            max_length=None,
            do_eval=False,
            ent_token="<ent>",
            use_cache=False
    ):
        """Load the data samples from file and tokenize the datasets"""
        source, domain = dataset_config.split("_")[0], dataset_config.split("_")[1]

        data_path = \
            f"{self.path}/{'wikipedia_silver' if source == 'wiki' else 'news_gold_test'}/en/{domain}/SPLIT.jsonl"
        if source == 'wiki':
            splits = ["valid", "test"] if do_eval else ["train", "valid", "test"]
        else:
            splits = ["test"]

        dataset = {}
        for sp in splits:
            cache_path = \
                f"{self.cache_dir}/cache_{'wikipedia_silver' if source == 'wiki' else 'news_gold_test'}_{domain}_{sp}_ent{str(add_ent_tokens)}_desc{str(add_desc_tokens)}_fact{str(add_fact)}_{type(tokenizer).__name__}"

            if cache_path and os.path.isfile(cache_path) and use_cache:
                print(f"Load tokenized data from cache at {cache_path}")
                split_data = torch.load(cache_path)
            else:
                print(f"Load data from {data_path.replace('SPLIT', sp)}")
                split_data = self.load_data_from_file(data_path.replace("SPLIT", sp))
                print("Tokenize and encode data")
                split_data = self.preprocess_data(split_data, 
                                                  tokenizer, 
                                                  add_ent_tokens=add_ent_tokens, 
                                                  add_desc_tokens=add_desc_tokens, 
                                                  add_fact=add_fact, 
                                                  max_length=max_length, 
                                                  ent_token=ent_token)
                torch.save(split_data, cache_path)  # cache the tokenized and encoded data

            dataset[sp] = split_data
        if "train" not in dataset:
            dataset["train"] = None
        if "valid" not in dataset:
            dataset["valid"] = None

        return dataset
