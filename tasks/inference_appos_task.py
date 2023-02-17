import logging
import os
import sys
import json
from dataclasses import dataclass, field
from typing import Optional
from tqdm import tqdm
from datasets import load_metric
from torch.utils.data import DataLoader

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

from src.data.collators import CollatorWithTrimForApposGen
from src.data.appos import ApposProcessor
from src.modules.evaluator import F1Evaluator

logger = logging.getLogger(__name__)



def print_for_paste(results):
    lines = []
    for k in ["Full", "Non-Empty", "Constraint"]:
        v = results[k]
        lines.append((k, v["f1"], v["meteor"], v["bleu"]))
    
    for line in lines:
        print(" & ".join([str(i) for i in line]))

def get_constraint_id(data):
    constraint_idxs = set()

    for idx, sample in tqdm(enumerate(data), total=len(data), desc="Extract contraint indexs"):
        if True in sample["used"]:
            constraint_idxs.add(idx)
    return constraint_idxs

def get_empty_id(data):
    empty_idxs = set()

    for idx, sample in tqdm(enumerate(data), total=len(data), desc="Extract empty indexs"):
        if len(sample["target"]) == 0:
            empty_idxs.add(idx)
    return empty_idxs


def evaluate(generations, targets, constraint_idxs, empty_idxs, metric_f1, metric_meteor, metric_bleu):
    results = {}

    preds = generations
    refs = [[i] for i in targets]

    blue_preds = [i.split(" ") for i in preds]
    bleu_refs = [[i.split(" ")] for i in targets]

    result_f1 = metric_f1.compute(predictions=preds, references=refs)
    result_meteor = metric_meteor.compute(predictions=preds, references=refs)
    result_bleu = metric_bleu.compute(predictions=blue_preds, references=bleu_refs)
    results["Full"] = {
        "f1": round(result_f1['f1'], 2),
        'meteor': round(result_meteor['meteor'], 3),
        'bleu': round(result_bleu['bleu'], 3),
    }


    preds = [i for idx, i in enumerate(generations) if idx not in empty_idxs]
    refs = [[i] for idx, i in enumerate(targets) if idx not in empty_idxs]

    blue_preds = [i.split(" ") for i in preds]
    bleu_refs = [[i.split(" ")] for idx, i in enumerate(targets) if idx not in empty_idxs]

    result_f1 = metric_f1.compute(predictions=preds, references=refs)
    result_meteor = metric_meteor.compute(predictions=preds, references=refs)
    result_bleu = metric_bleu.compute(predictions=blue_preds, references=bleu_refs)
    results["Non-Empty"] = {
        "f1": round(result_f1['f1'], 2),
        'meteor': round(result_meteor['meteor'], 3),
        'bleu': round(result_bleu['bleu'], 3),
    }



    preds = [i for idx, i in enumerate(generations) if idx in constraint_idxs]
    refs = [[i] for idx, i in enumerate(targets) if idx in constraint_idxs]


    blue_preds = [i.split(" ") for i in preds]
    bleu_refs = [[i.split(" ")] for idx, i in enumerate(targets) if idx in constraint_idxs]

    result_f1 = metric_f1.compute(predictions=preds, references=refs)
    result_meteor = metric_meteor.compute(predictions=preds, references=refs)
    result_bleu = metric_bleu.compute(predictions=blue_preds, references=bleu_refs)
    results["Constraint"] = {
        "f1": round(result_f1['f1'], 2),
        'meteor': round(result_meteor['meteor'], 3),
        'bleu': round(result_bleu['bleu'], 3),
    }

    print("======== Results ========")
    print("Type & F1 & Meteor & BLEU")
    print_for_paste(results)


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
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
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
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
            "the model's position embeddings."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    data_dir: Optional[str] = field(
        default="/data/appos_corpus", metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    data_cache_dir: Optional[str] = field(
        default="/data/cache/appos_corpus/",
        metadata={"help": "Where to store the pretokenized data"},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
            "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
            "during ``evaluate`` and ``predict``."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
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
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default=None, metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )
    data_cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the data"},
    )

    eval_beams: Optional[int] = field(default=None, metadata={"help": "# num_beams to use for evaluation."})
    split: Optional[str] = field(
        default=None, metadata={"help": ""}
    )
    add_ent_tokens: bool = field(
        default=False, metadata={"help": ""}
    )
    add_desc_tokens: bool = field(
        default=False, metadata={"help": ""}
    )
    add_fact: bool = field(
        default=False, metadata={"help": ""}
    )




def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
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

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        add_prefix_space=True,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    model.resize_token_embeddings(len(tokenizer))

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")


    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    # Temporarily set max_source_length for training.
    max_source_length = data_args.max_source_length
    # whether we should load the data cache or not: for kelm yes; bart no
    use_cache = False if model_args.model_name_or_path == "facebook/bart-base" or model_args.model_name_or_path == "facebook/bart-large" else True
    
    # Load and Preprocess the dataset
    ent_token = "<ap>" if "_ap" in model_args.model_name_or_path else "<ent>"
    processor = ApposProcessor(data_path=data_args.data_dir, cache_dir=data_args.data_cache_dir)
    appos_datasets = processor.load_dataset(
        data_args.dataset_config_name, 
        tokenizer, add_ent_tokens=data_args.add_ent_tokens, 
        add_desc_tokens=data_args.add_desc_tokens, 
        add_fact=data_args.add_fact, 
        max_length=max_source_length, 
        do_eval=True, 
        ent_token=ent_token, 
        use_cache=use_cache
    )

    # Data collator
    data_collator = CollatorWithTrimForApposGen(tokenizer.pad_token_id, model, inference=True)

    # Metric
    metric_f1 = F1Evaluator()
    metric_meteor = load_metric("meteor")
    metric_bleu= load_metric("bleu")


    # use data args for generation
    gen_params = {"num_beams": data_args.eval_beams if data_args.eval_beams is not None else model.config.num_beams}
    model.config.update(gen_params)
    model = model.cuda()
    

    # Debugging
    data = appos_datasets[data_args.split]
    if data is None:
        raise ValueError(f"Split {data_args.split} is None.")
    loader = DataLoader(data, batch_size=1, collate_fn=data_collator)

    descriptions = []
    targets = []
    generations = []
    generations_full = []
    for batch in tqdm(enumerate(loader), total=len(loader)):

        label_masks = batch["label_mask"].tolist()[0]
        generation_idx = label_masks.index(1)
        right_len = len(label_masks) - generation_idx - sum(label_masks)
        seq = model.generate(
            batch["input_ids"].cuda(),
            decoder_input_ids=batch["decoder_input_ids"][:,:generation_idx].cuda(),
            max_length=generation_idx + data_args.max_target_length,
            eos_token_id=tokenizer.eos_token_id
        )
        gen_text_full = tokenizer.batch_decode(seq[:, generation_idx:], clean_up_tokenization_spaces=True)

        decoder_input_ids_left = tokenizer.batch_decode(
            batch["decoder_input_ids"][:, -right_len+1:],
            clean_up_tokenization_spaces=True
        )[0][:10]

        try:
            end_index = gen_text_full[0].index(decoder_input_ids_left)
            gen_text = gen_text_full[0][:end_index]
        except:
            gen_text = tokenizer.batch_decode(seq[:, generation_idx:-right_len], clean_up_tokenization_spaces=True)[0]

        # Extract the labels from the sequences
        v = batch["labels"]
        v[v==-100] = tokenizer.pad_token_id
        gold = tokenizer.batch_decode(v, skip_special_tokens=False, clean_up_tokenization_spaces=True)

        # Extract the description labels from the sequences
        v = batch["desc_labels"]
        v[v==-100] = tokenizer.pad_token_id
        desc = tokenizer.batch_decode(v, skip_special_tokens=False, clean_up_tokenization_spaces=True)

        targets.append(gold[0].replace("<pad>", "").replace("<s>", "").replace("</s>", "").strip())
        descriptions.append(desc[0].replace("<pad>", "").replace("<s>", "").replace("</s>", "").strip())
        generations.append(gen_text.replace("</ent_desc>", "").strip())
        generations_full.append(gen_text_full[0])

    os.makedirs(training_args.output_dir, exist_ok=True)

    with open(f"{training_args.output_dir}/{data_args.dataset_config_name}_targets_{data_args.split}.txt", "w") as f:
        for line in targets:
            f.write(line.replace("\n", " ") + "\n")
    
    with open(
            f"{training_args.output_dir}/{data_args.dataset_config_name}_descriptions_{data_args.split}.txt", "w"
    ) as f:
        for line in descriptions:
            f.write(line.replace("\n", " ") + "\n")

    with open(
            f"{training_args.output_dir}/{data_args.dataset_config_name}_generations_{data_args.split}.txt", "w"
    ) as f:
        for line in generations:
            f.write(line.replace("\n", " ") + "\n")
    
    with open(
            f"{training_args.output_dir}/{data_args.dataset_config_name}_generations_full_{data_args.split}.txt", "w"
    ) as f:
        for line in generations_full:
            f.write(line.replace("\n", " ") + "\n")

    # Post-process the generations and the targets for evaluation
    print(f'Load {training_args.output_dir}/{data_args.dataset_config_name}_targets_{data_args.split}.txt')
    with open(f"{training_args.output_dir}/{data_args.dataset_config_name}_targets_{data_args.split}.txt", "r") as f:
        targets = f.readlines()
    
    new_targets = []
    for i in targets:
        i = i.replace("<s>", "").replace("</s>", "").strip()
        if len(i) == 0:
            new_targets.append("<EMPTY>")
        else:
            new_targets.append(i)
    targets = new_targets

    print(f'Load {training_args.output_dir}/{data_args.dataset_config_name}_generations_{data_args.split}.txt')
    with open(
            f"{training_args.output_dir}/{data_args.dataset_config_name}_generations_{data_args.split}.txt", "r"
    ) as f:
        generations = f.readlines()
    
    new_generations = []
    for i in generations:
        i = i.replace("<s>", "").replace("</s>", "").strip()
        if len(i) == 0:
            new_generations.append("<EMPTY>")
        else:
            new_generations.append(i)
    generations = new_generations

    if "<sep>" in generations[0]:
        generations = [line.split("<sep>")[-1].strip() for line in generations]

    source, domain = data_args.dataset_config_name.split("_")[0], data_args.dataset_config_name.split("_")[1]

    data_path = \
        f"{data_args.data_dir}/{'wikipedia_silver' if source == 'wiki' else 'news_gold_test'}/en/{domain}/SPLIT.jsonl"

    with open(data_path.replace("SPLIT", data_args.split), "r") as f:
        data = []
        for line in f.readlines():
            data.append(json.loads(line))

    # Extract the indexes for the data samples in the "constraint" type
    constraint_idxs = get_constraint_id(data)
    # Extract the indexes for the data samples in the "empty" type
    empty_idxs = get_empty_id(data)
    # Evaluate the generations
    evaluate(generations, targets, constraint_idxs, empty_idxs, metric_f1, metric_meteor, metric_bleu)


if __name__ == "__main__":
    main()
