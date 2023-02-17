import logging
import os
import sys
import math
import json
from dataclasses import dataclass, field
from typing import Optional

import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    BartConfig,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

import datasets
from datasets import (
    load_dataset, 
    load_from_disk
)

from src.data.collators import DataCollatorForSeq2Seq
from src.data.wiki_data_utils import proc_wiki_dataset
from src.model.bart_modeling import BartForConditionalGenerationWithDesc
from src.utils.utils import (
    save_json,
    check_output_dir,
)
from src.modules.trainer import NewSeq2SeqTrainer
from src.modules.evaluator import (
    handle_metrics,
)


logger = logging.getLogger(__name__)

# Add special tokens
ATTR_TO_SPECIAL_TOKEN = {'pad_token': '<pad>', 'mask_token': '<mask>',
                         'additional_special_tokens': ['<ent>', '</ent>', '<ent_desc>', '</ent_desc>', "<sep>"]}


def add_special_tokens_(tokenizer, model):
    """ Add special tokens to the tokenizer and the model if they have not already been added. """
    orig_num_tokens = len(tokenizer) #len(tokenizer.encoder)
    num_added_tokens = tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN) # doesn't add if they are already there
    if num_added_tokens > 0:
        model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens)


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
    overwrite_tokenize: bool = field(
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
        default=3.0, metadata={"help": "A number of text spans are sampled, with span lengths drawn from a Poisson distribution (λ = 3)."}
    )
    mask_ent_probability: float = field(
        default=0.3, metadata={"help": "The probability that an entity in the paragraphs being masked."}
    )
    eval_beams: Optional[int] = field(default=None, metadata={"help": "# num_beams to use for evaluation."})
    pad_to_multiple_of: Optional[int] = field(default=8, metadata={"help": "Multiple times of # tokens will the sequence be padded to"})
    train_w_no_duplicate: bool = field(
        default=False, metadata={"help": "Train the model with no duplicated entities."}
    )

@dataclass
class ExtraArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    separate_loss: bool = field(
        default=False,
        metadata={"help": "Separate pre-training loss and short description loss"},
    )
    entity_weight: Optional[float] = field(
        default=1.0,
        metadata={"help": "The weight that the entity part takes in the pre-training"},
    )


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments, ExtraArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, extra_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, extra_args = parser.parse_args_into_dataclasses()

    check_output_dir(training_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed
    set_seed(training_args.seed)

    # Check overwrite status
    if data_args.overwrite_cache:
        data_args.overwrite_tokenize = True

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

    # add extra configurations for training options
    config.separate_loss = extra_args.separate_loss
    config.entity_weight = extra_args.entity_weight
    model = BartForConditionalGenerationWithDesc.from_pretrained(
        model_args.model_name_or_path,
        from_tf=".ckpt" in model_args.model_name_or_path,
        config=config,
    )

    # add special tokens
    add_special_tokens_(tokenizer, model)
    model.config.vocab_size = len(tokenizer)

    # use data args for generation
    gen_params = {
        "num_beams": data_args.eval_beams if data_args.eval_beams is not None else model.config.num_beams,
        "max_length": data_args.max_target_length
    }
    model.config.update(gen_params)

    # set num_beams for evaluation
    if data_args.eval_beams is None:
        data_args.eval_beams = model.config.num_beams

    dataset_dir = f"{data_args.cache_dir}/wikipedia/mid2-{data_args.dataset_config_name}-bart-{data_args.mlm_probability}-{data_args.poisson_lambda}"

    if (training_args.do_train and os.path.exists(dataset_dir+"/train")) or (training_args.do_eval and os.path.exists(dataset_dir+"/valid")) or (training_args.do_predict and os.path.exists(dataset_dir+"/test")):
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
        train_dataset, eval_dataset, test_dataset = proc_wiki_dataset[data_args.task](training_args, data_args, lm_datasets, dataset_dir, tokenizer, epochs=int(training_args.num_train_epochs),\
                                                                                      mlm_probability=data_args.mlm_probability, poisson_lambda=data_args.poisson_lambda)
    

    # change number of samples to be 1, since we alreay expand the data
    training_args.num_train_epochs = 1.0
    if data_args.max_train_samples is not None:
        train_dataset = train_dataset.select(range(data_args.max_train_samples))


    # Initialize the data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, max_length=data_args.max_source_length)

    # Initialize the trainer
    trainer = NewSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    all_metrics = {}
    # Training
    if training_args.do_train:
        logger.info("*** Train ***")
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics

        trainer.save_model() 

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        try:
            perplexity = math.exp(metrics["train_loss"])
        except OverflowError:
            perplexity = float("inf")

        metrics["train_loss"] = round(metrics["train_loss"], 4)
        metrics["train_ppl"] = perplexity

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        if trainer.is_world_process_zero():
            handle_metrics("train", metrics, training_args.output_dir)
            all_metrics.update(metrics)

            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation on valid seen set
    save_appendix = f"mlm{str(data_args.mlm_probability)}-poisson{str(data_args.poisson_lambda)}"
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate(
            metric_key_prefix="eval", max_length=data_args.max_target_length, num_beams=data_args.eval_beams
        )
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        
        if extra_args.separate_loss:
            metrics[f"eval-m-{save_appendix}_loss"] = round(metrics["eval_loss"], 4)
            metrics[f"eval-m-{save_appendix}_ppl"] = perplexity

            trainer.log_metrics(f"eval-m-{save_appendix}", metrics)
            trainer.save_metrics(f"eval-m-{save_appendix}", metrics)
        else:
            metrics[f"eval-{save_appendix}_loss"] = round(metrics["eval_loss"], 4)
            metrics[f"eval-{save_appendix}_ppl"] = perplexity

            trainer.log_metrics(f"eval-{save_appendix}", metrics)
            trainer.save_metrics(f"eval-{save_appendix}", metrics)

        if trainer.is_world_process_zero():
            handle_metrics("val", metrics, training_args.output_dir)
            all_metrics.update(metrics)
    
    # Evaluation on valid unseen set
    if training_args.do_predict:
        logger.info("*** TEST ***")

        metrics = trainer.evaluate(
            eval_dataset=test_dataset, metric_key_prefix="test", max_length=data_args.max_target_length, num_beams=data_args.eval_beams
        )
        max_test_samples = data_args.max_test_samples if data_args.max_test_samples is not None else len(eval_dataset)
        metrics["test_samples"] = min(max_test_samples, len(test_dataset))
        try:
            perplexity = math.exp(metrics["test_loss"])
        except OverflowError:
            perplexity = float("inf")

        if extra_args.separate_loss:
            metrics[f"test-m-{save_appendix}_loss"] = round(metrics["test_loss"], 4)
            metrics[f"test-m-{save_appendix}_ppl"] = perplexity

            trainer.log_metrics(f"test-m-{save_appendix}", metrics)
            trainer.save_metrics(f"test-m-{save_appendix}", metrics)
        else:
            metrics[f"test-{save_appendix}_loss"] = round(metrics["test_loss"], 4)
            metrics[f"test-{save_appendix}_ppl"] = perplexity

            trainer.log_metrics(f"test-{save_appendix}", metrics)
            trainer.save_metrics(f"test-{save_appendix}", metrics)

        if trainer.is_world_process_zero():
            handle_metrics("test", metrics, training_args.output_dir)
            all_metrics.update(metrics)

    # Save all the results
    if trainer.is_world_process_zero():
        save_json(all_metrics, os.path.join(training_args.output_dir, "all_results.json"))

    return all_metrics


if __name__ == "__main__":
    main()