# Introduction
[KILM](<TODO: LINK TO ARCHIVE PAPER>) (short for Knowledgeable
Injection into Language Model) is an approach for injecting knowledge into
pre-trained language models. This repository provides 
the necessary code for KILM.
KILM propose a new (second-step) pre-trianing method to inject information
about entities (such as entity descriptions) into pre-trained language models. 


# KILM Code Base

This repository contains code for  

1) Creating the dataset needed for model training. The code 
goes through a dump of Wikipedia and extracts the necessary data 
(short descriptions, etc.).
2) Language model continued pre-training (knowledge injection)
3) Evaluation of the final model on some downstream-tasks. 


In the following sections, we describe the dependencies, the steps to 
reproduce our pre-training checkpoint, the scripts for downstream task 
evaluation, and the code structure illustration.

## Details

KILM relies on a special data structure for short description knowledge 
injection. All the experiments are conducted based on the BART model. 
To maintain the language modeling ability of the original BART, we still 
maintain `Text Infilling` as one of our pre-training objectives. To make the 
special data structure, we introduce five more extra special tokens: `<ent>`, 
`</ent>`, `<ent_desc>`, `</ent_desc>`, and `<sep>`, and combine a sentence 
with the short description of one entity in the sentence as follows:


    "Andrew R. Jassy is an American business executive who has been the president and 
    <ent> CEO </ent><ent_desc>Cheif Executive Officer <s> Highest-ranking corporate 
    officer or administrator</ent_desc> of Amazon since July 5, 2021 ."


Similar to BART, we mask around 30% of the tokens with `<mask>` token, where 
a span of tokens will be replaced with only one `<mask>` token. Besides that, 
we also mask the whole short description with one `<mask>` token. During pre-trianing, 
the model learns to recover the `<mask>` tokens into their original surface 
tokens.


To reproduce the existing checkpoint, please first install the dependencies 
and go through the following steps:

## General Structure of the Code 
```
kilm
├── data                                    : Contains the code for data pre-processing
│   └── wiki
│       ├── util.py                         : Utility functions for data pre-processing Task 1
│       ├── wiki_short_desc.py              : Main functions for data pre-processing Task 1
│       └── wiki_preprocess.py              : Main functions for data pre-processing Tasks 2, 3, 4
├── scripts                                 : Contains command line scripts
│   └── run_pretraining.sh                  : Train/eval script for KILM
│   └── pretraining_inference.sh            : Running inference on pre-trained models for short description generation
├── src                                     : Contains all the functions for training/finetuning KILM
│   ├── data                                : Contains the code for data loading/encoding/collating
│   │   ├── __init__.py
│   │   ├── appos.py                        : Data loading/encoding functions for appositive generation
│   │   ├── collators.py                    : Data collators that support "trim batch" or label padding
│   │   ├── wiki_data_utils.py              : Data encoding functions for KILM pre-training
│   │   └── wikipedia.py                    : Data loading function for Wikipedia articles and short descriptions
│   ├── model                               : Contains the model-related code
│   │   ├── __init__.py
│   │   └── bart_modeling.py                : Bart model code, modified from huggingface to have more inputs
│   ├── modules                             : Contains the code for some modified HF classes
│   │   ├── __init__.py
│   │   ├── evaluator.py                    : Evaluation function for unigram-F1, ROUGE, BLEU
│   │   ├── retriever.py                    : TFIDF and random retriever class for few-shot learning on Natural Questions dataset
│   │   └── trainer.py                      : Modified Trainer class that avoids saving the optimizer weights for checkpoints (OOM)
│   └── utils                               : Contains some utility functions
│       ├── __init__.py
│       ├── utils_nq.py                     : Code for converting Natural Questions data sample structure into a simpler version (copy from NQ github)
│       └── utils.py                        : Random utility functions
├── tasks                                   : Contains code for running pretraining, inference and downstream tasks
│       ├── pretraining.py                  : Main function for KILM pre-training
│       ├── inference.py                    : Inference code for KILM pre-training to have the model generate short descriptions for any input text
│       ├── infer_appos_generation.py       : Inference for appositive generation with finetune model / zero-shot evaluation
│       └── infer_nq_task.py                : Zero/few-shot evaluation on Natural Questions dataset 
└── 
```


## Dependencies

The required dependencies could be installed by running the following:
```console
pip install -r requirements.txt
```

## Data Pre-processing

The wikipedia dump is publicly available [here](https://dumps.wikimedia.org/enwiki/). 
After downloading the wikipedia dump, please go to folder `data/wiki` for data 
pre-processing.

The data pre-processing includes:


* `Step 1`: Convert the Wikipedia dump from `XML` format to `JSON` format, along with 
    data cleansing and short description extraction;
* `Step 2`: Data filtering on Wikipedia articles to filter out those samples that are 
    not suitable for pre-training; (for simplicity, we only keep the summary 
    part of the Wikipedia page for now)
* `Step 3`: Data filtering on short descriptions to filter out those from wikipedia 
    disambiguation page;
* `Step 4`: Train/Valid seen/Valid unseen set split;

Please refer to `data/wiki/README.md` for more details. The following is how to run these steps:

```console
cd data/wiki

# Step 1
python tasks/wiki_short_desc.py --wiki_dump <address to the downloaded dump of wikipedia> --out_dir <a directory for output>

# Steps 2 and 3
python tasks/wiki_preprocess.py --task filter --data_folder <address to the wikipedia folder>

# Step 4
python tasks/wiki_preprocess.py --task split --data_folder <address to the wikipedia folder>
```


## Pre-training

Now, we can start the (second step of) pre-training process:

```console
bash scripts/run_pretraining.sh <output folder to save the model>
```

## Inference

A direct advantage of KILM is that it is able to generate the short
description of an entity, following the data structure that the model is pre-trained
on. 

The corresponding scripts for running the LM inference are in `script` folder, 
```console
bash scripts/test_pretraining_inference.sh <path to the model> <path to the tokenized data folder>
```


## Evaluation on Downstream Tasks
The pre-trained checkpoints are further evaluated on downstream tasks, 
including: GLUE, SuperGLUE, Entity Linking, WoW, Summarization, Appositive Generation, and QA.
The repository provides code for running Natural Questions
and appositive generation experiments. For other tasks
we provide links to existing publicly available code.
We conduct experiments on GLUE, SuperGLUE, and Summarization task 
with the code bases from HuggingFace examples [text classification](https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue.py)
and [summarization](https://github.com/huggingface/transformers/blob/main/examples/pytorch/summarization/run_summarization.py).
Experiments on entity linking tasks are done using the  BLINK 
[code base](https://github.com/facebookresearch/BLINK).

### Appositive Generation

We conduct zero-shot evaluation on Appostive Generation task. The dataset 
ApposCorpus is involved. There are four settings from the dataset: 
`news_org`, `news_per`, `wiki_org`, `wiki_per`. We conduct different 
methods for evaluating BART and KILM.

* To evaluate KILM model:

```console
python tasks/inference_appos_task.py \
--model_name_or_path <path to the model> \
--dataset_config_name <news_org/news_per/wiki_org/wiki_per> \
--split test \
--output_dir <folder name where the generation results are saved> \
--max_source_length 512 \
--pad_to_max_length \
--data_dir <PATH TO PREPARED DATA>  \
--data_cache_dir <PATH TO WHERE DATA SHOULD BE CACHED> \
--cache_dir <PATH TO TRANSFORMER CACHE DIR> \
--overwrite_output_dir \
--add_ent_tokens \
--add_desc_tokens \
--lm_probing \
--topk 0 \
--max_source_length 824
```

### QA

We evaluate our model and BART-base/large models under zero/few-shot settings for QA datasets.

In this work, three QA datasets are involved: 
* Natural Questions (NQ)
* WebQuestions
* TriviaQA

In total, there're four ways of prompting, denoted as method "1"/"2"/"3"/"4".
Method "1" and "2" are using "Question"/"Answer" as the trigger words, while
method "3" and "4" are using "Q"/"A". The difference between "1" & "2", and,
"3" & "4" is whether there're space following the colon after the "Answer"/"A". 
This may result in different results on the zero/few-shot QA tasks, which may 
related to the tokenization process of the models.


#### Zero-shot settings

```console
python tasks/inference_qa_task.py \
--model_path <path to the model> \
--save_path <path to the save folder> \
--mode zero-shot \
--method <1/2/3/4> \
--dataset <nq/triviaqa/webquestions>
```

#### Few-shot settings

We consider 1/2/5/8 shot in this project. For retrieving the example QA pairs, 
TF-IDF retriever is leveraged

The TF-IDF retriever is implemented based on this [code base](https://github.com/efficientqa/retrieval-based-baselines#tfidf-retrieval).


```console
python tasks/inference_qa_task.py \
--model_path <path to the model> \
--save_path <path to the save folder> \
--mode <1/2/5/8-shot> \
--method <1/2/3/4> \
--dataset <nq/triviaqa/webquestions> \
--tfidf_path <path to the tfidf model> \
--db_path <path to the data file used when training the tfidf model>
```

# Citing KILM
To cite this work please use the following:
```console
Yan Xu, Mahdi Namazifar, Devamanyu Hazarika, Aishwarya Padmakumar, Yang Liu, Dilek Hakkanii-Tür; 
"KILM: Knowledge Injection into Encoder-Decoder Language Models", 10.48550/ARXIV.2302.09170



Bibtex:

@misc{https://doi.org/10.48550/arxiv.2302.09170,
  doi = {10.48550/ARXIV.2302.09170},
  
  url = {https://arxiv.org/abs/2302.09170},
  
  author = {Xu, Yan and Namazifar, Mahdi and Hazarika, Devamanyu and Padmakumar, Aishwarya and Liu, Yang and Hakkani-Tür, Dilek},
  
  keywords = {Computation and Language (cs.CL), Artificial Intelligence (cs.AI), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {KILM: Knowledge Injection into Encoder-Decoder Language Models},
  
  publisher = {arXiv},
  
  year = {2023},
  
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```  
