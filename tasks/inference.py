import logging
import os
import sys
import json
from dataclasses import dataclass, field
from typing import Optional
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    BartConfig,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
)

import datasets
from datasets import load_dataset, load_from_disk

from src.data.wiki_data_utils import proc_wiki_dataset
from src.data.collators import DataCollatorForSeq2SeqWithLeftPadding
from src.model.bart_modeling import BartForConditionalGenerationWithDesc

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task: Optional[str] = field(
        default=None,
        metadata={"help": "Task name, summarization (or summarization_{dataset} for pegasus) or translation"},
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store processed data"},
    )
    data_dir: Optional[str] = field(
        default="/dev/shm/data",
        metadata={"help": "Where do you want to store the data"},
    )
    max_source_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=640,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_gen_length: Optional[int] = field(
        default=640,
        metadata={
            "help": "The maximum generation length "
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    mlm_probability: float = field(
        default=0.3, metadata={"help": "We mask 30 percent of tokens in each document, and permute all sentences."}
    )
    poisson_lambda: float = field(
        default=3.0, metadata={"help": "A number of text spans are sampled, with span lengths drawn "
                                       "from a Poisson distribution (λ = 3)."}
    )
    mask_ent_probability: float = field(
        default=0.3, metadata={"help": "The probability that an entity in the paragraphs being masked."}
    )
    eval_beams: Optional[int] = field(default=None, metadata={"help": "# num_beams to use for evaluation."})
    pad_to_multiple_of: Optional[int] = field(default=8, metadata={"help": "Multiple times of # tokens will the "
                                                                           "sequence be padded to"})


@dataclass
class ExtraArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    add_prefix_to_decoder: bool = field(
        default=False,
        metadata={"help": "Add the sequence before descriptions"},
    )


def post_processing_and_save(generations, golds, tokenizer, output_dir, split, add_prefix_to_decoder=True):
    entities = []
    descriptions = []
    gold_entities = []
    gold_descriptions = []

    excpetion_idxs = []
    
    for idx, (t, g) in enumerate(zip(generations, golds)):
        # Remove padding tokens if any
        _g = g.replace(tokenizer.pad_token, "").strip()
        if len(_g) != 0:
            if not add_prefix_to_decoder:
                t = t.split("<ent_desc>")[1]
            
            try:
                entity, t = t.split("<sep>")[0], t.split("<sep>")[1]
            except:
                entity = ""
                
            gold_entity,g = g.split("<sep>")[0], g.split("<sep>")[1]
            description = t.split("</ent_desc>")[0].strip() 
            gold_description = g.split("</ent_desc>")[0].strip()

            entities.append(entity.replace("\n", ""))
            descriptions.append(description.replace("\n", ""))
            gold_entities.append(gold_entity.replace("\n", ""))
            gold_descriptions.append(gold_description.replace("\n", ""))
            
        else:
            excpetion_idxs.append(idx)
    
    print("excpetion_idxs", excpetion_idxs)

    assert len(entities) == len(descriptions) == len(gold_entities) == len(gold_descriptions)

    with open(os.path.join(output_dir, f"{split}_prefix-{str(add_prefix_to_decoder)}_gen_entity.txt"), "w") as f:
        for line in entities:
            f.write(line + "\n")
    
    with open(os.path.join(output_dir, f"{split}_prefix-{str(add_prefix_to_decoder)}_gen_description.txt"), "w") as f:
        for line in descriptions:
            f.write(line + "\n")
    
    with open(os.path.join(output_dir, f"{split}_prefix-{str(add_prefix_to_decoder)}_gold_entity.txt"), "w") as f:
        for line in gold_entities:
            f.write(line + "\n")
    
    with open(os.path.join(output_dir, f"{split}_prefix-{str(add_prefix_to_decoder)}_gold_description.txt"), "w") as f:
        for line in gold_descriptions:
            f.write(line + "\n")



def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments, ExtraArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, extra_args = \
            parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, extra_args = parser.parse_args_into_dataclasses()

    # Set seed
    set_seed(training_args.seed)

    # Check overwrite status

    # Load pretrained model and tokenizer
    config_kwargs = {
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        **config_kwargs,
    )

    tokenizer_kwargs = {
        "use_fast": model_args.use_fast_tokenizer,
        "add_prefix_space": True if isinstance(config, BartConfig) else False,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        **tokenizer_kwargs,
    )

    model = BartForConditionalGenerationWithDesc.from_pretrained(
        model_args.model_name_or_path,
        from_tf=".ckpt" in model_args.model_name_or_path,
        config=config,
    )

    # use data args for generation
    gen_params = {"num_beams": data_args.eval_beams if data_args.eval_beams is not None else model.config.num_beams,
                  "max_length": data_args.max_target_length}
    model.config.update(gen_params)
    model = model.cuda()

    # set num_beams for evaluation
    if data_args.eval_beams is None:
        data_args.eval_beams = model.config.num_beams

    # Load the datasets
    dataset_dir = f"{data_args.cache_dir}/wikipedia/mid2-{data_args.dataset_config_name}-bart-{data_args.mlm_probability}-{data_args.poisson_lambda}-genTrue"
    if (training_args.do_train and os.path.exists(dataset_dir+"/train")) or \
            (training_args.do_eval and os.path.exists(dataset_dir+"/valid")) or \
            (training_args.do_predict and os.path.exists(dataset_dir+"/test")
            ):
        logger.info("*** Load dataset from disk ***")
        if training_args.do_train:
            train_dataset = load_from_disk(dataset_dir+"/train")
        else:
            train_dataset = None
        if training_args.do_eval:
            eval_dataset = load_from_disk(dataset_dir+"/valid")
        else:
            eval_dataset = None
        if training_args.do_predict:
            test_dataset = load_from_disk(dataset_dir+"/test")
        else:
            test_dataset = None
    else:
        logger.info("*** Prepare dataset features from scratch ***")
        lm_datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name, cache_dir=data_args.cache_dir)
        train_dataset, eval_dataset, test_dataset = \
            proc_wiki_dataset[data_args.task](
                training_args,
                data_args,
                lm_datasets,
                dataset_dir,
                tokenizer,
                epochs=int(training_args.num_train_epochs),
                mlm_probability=data_args.mlm_probability,
                poisson_lambda=data_args.poisson_lambda,
                generate=True
            )
    
    # Make folders for saving the outputs
    os.makedirs(training_args.output_dir, exist_ok = True) 
    # Prepare data collator
    data_collator = DataCollatorForSeq2SeqWithLeftPadding(
        tokenizer=tokenizer, model=model, max_length=data_args.max_source_length, trigger_length=1
    )

    # Inference on Valid seen set
    if training_args.do_eval:
        eval_loader = DataLoader(
            eval_dataset, batch_size=training_args.per_device_eval_batch_size, collate_fn=data_collator
        )

        if not os.path.exists(os.path.join(
                training_args.output_dir,
                f"eval_prefix-{str(extra_args.add_prefix_to_decoder)}_generations.json"
        )) or training_args.overwrite_output_dir:
            generations = []
            golds = []

            # Generate batch-by-batch
            for batch in tqdm(eval_loader, total=len(eval_loader)):
                if extra_args.add_prefix_to_decoder:
                    orig_length = batch["decoder_input_ids"].size(1)
                    seq = model.generate(
                        batch["input_ids"].cuda(),
                        decoder_input_ids=batch["decoder_input_ids"].cuda(),
                        max_length=data_args.max_gen_length,
                        eos_token_id=tokenizer.convert_tokens_to_ids("</ent_desc>")
                    )
                else:
                    orig_length = 0
                    seq = model.generate(
                        batch["input_ids"].cuda(),
                        max_length=data_args.max_target_length,
                        eos_token_id=tokenizer.convert_tokens_to_ids("</ent_desc>")
                    )

                text = tokenizer.batch_decode(seq[:, orig_length:], skip_special_tokens=False)
                gold_text = tokenizer.batch_decode(batch["label_mask"], skip_special_tokens=False)

                generations.extend(text)
                golds.extend(gold_text)

            # Save the generations into files
            with open(os.path.join(
                    training_args.output_dir,
                    f"eval_prefix-{str(extra_args.add_prefix_to_decoder)}_generations.json"), "w") as f:
                for line in generations:
                    json.dump(line, f)
                    f.write("\n")
            
            with open(os.path.join(
                    training_args.output_dir,
                    f"eval_prefix-{str(extra_args.add_prefix_to_decoder)}_golds.json"), "w") as f:
                for line in golds:
                    json.dump(line, f)
                    f.write("\n")

        """Post-process the generations"""
        with open(os.path.join(
                training_args.output_dir,
                f"eval_prefix-{str(extra_args.add_prefix_to_decoder)}_generations.json"), "r") as f:
            lines = f.readlines()
            generations = []
            for line in lines:
                generations.append(json.loads(line))
        
        with open(os.path.join(
                training_args.output_dir,
                f"eval_prefix-{str(extra_args.add_prefix_to_decoder)}_golds.json"), "r") as f:
            lines = f.readlines()
            golds = []
            for line in lines:
                golds.append(json.loads(line))

        # Parse the generations and re-save them into files
        post_processing_and_save(
            generations,
            golds,
            tokenizer,
            training_args.output_dir,
            "eval",
            add_prefix_to_decoder=extra_args.add_prefix_to_decoder
        )

    # Inference on Valid unseen set
    if training_args.do_predict:
        test_loader = DataLoader(
            test_dataset,
            batch_size=training_args.per_device_eval_batch_size,
            collate_fn=data_collator
        )

        if not os.path.exists(os.path.join(
                training_args.output_dir,
                f"test_prefix-{str(extra_args.add_prefix_to_decoder)}_generations.json")
        ) or training_args.overwrite_output_dir:
            generations = []
            golds = []

            # Generate batch-by-batch
            for batch in tqdm(test_loader, total=len(test_loader)):
                if extra_args.add_prefix_to_decoder:
                    orig_length = batch["decoder_input_ids"].size(1)
                    seq = model.generate(
                        batch["input_ids"].cuda(),
                        decoder_input_ids=batch["decoder_input_ids"].cuda(),
                        max_length=data_args.max_gen_length,
                        eos_token_id=tokenizer.convert_tokens_to_ids("</ent_desc>")
                    )
                else:
                    orig_length = 0
                    seq = model.generate(
                        batch["input_ids"].cuda(),
                        max_length=data_args.max_target_length,
                        eos_token_id=tokenizer.convert_tokens_to_ids("</ent_desc>")
                    )
                    
                text = tokenizer.batch_decode(seq[:, orig_length:], skip_special_tokens=False)
                gold_text = tokenizer.batch_decode(batch["label_mask"], skip_special_tokens=False)

                generations.extend(text)
                golds.extend(gold_text)
            
            # Save the generations into files
            with open(os.path.join(
                    training_args.output_dir,
                    f"test_prefix-{str(extra_args.add_prefix_to_decoder)}_generations.json"), "w") as f:
                for line in generations:
                    json.dump(line, f)
                    f.write("\n")
            
            with open(os.path.join(
                    training_args.output_dir,
                    f"test_prefix-{str(extra_args.add_prefix_to_decoder)}_golds.json"), "w") as f:
                for line in golds:
                    json.dump(line, f)
                    f.write("\n")

        """Post-process the generations"""
        with open(os.path.join(
                training_args.output_dir,
                f"test_prefix-{str(extra_args.add_prefix_to_decoder)}_generations.json"), "r") as f:
            lines = f.readlines()
            generations = []
            for line in lines:
                generations.append(json.loads(line))
        
        with open(os.path.join(
                training_args.output_dir,
                f"test_prefix-{str(extra_args.add_prefix_to_decoder)}_golds.json"), "r") as f:
            lines = f.readlines()
            golds = []
            for line in lines:
                golds.append(json.loads(line))

        # Parse the generations and re-save them into files
        post_processing_and_save(
            generations,
            golds,
            tokenizer,
            training_args.output_dir,
            "test",
            add_prefix_to_decoder=extra_args.add_prefix_to_decoder
        )


if __name__ == "__main__":
    main()