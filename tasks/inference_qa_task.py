import os
import sys
import json
import time
import argparse
from gzip import GzipFile
from tqdm import tqdm
from collections import defaultdict
from datasets import load_dataset

import torch
import transformers 
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


from src.utils.utils_nq import simplify_nq_example
from src.modules.retriever import TfidfRetriever, RandomRetriever
from src.modules.evaluator import F1Evaluator, normalize_answer


def postprocess_text(text):
    text = text.split("\n")[0].split("Question:")[0]
    return text


def convert_webquestions_to_nq_format(data):
    samples = []
    for sample in data:
        question = sample["question"]
        answer = sample["answers"]
        samples.append({"question_text": question, "answers": answer})
    print("Data Example:", samples[0])
    return samples


def convert_triviaqa_to_nq_format(data):
    samples = []
    for sample in data:
        question = sample["question"]
        answer = sample['answer']['normalized_aliases']
        samples.append({"question_text": question, "answers": answer})

    print("Data Example:", samples[0])
    return samples


argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--model_path",
    default=None,
    type=str,
    help="Full path to the model for evaluation")
argparser.add_argument(
    "--save_path",
    default=None,
    type=str,
    help="Folder to save the generation results")
argparser.add_argument(
    "--mode",
    default="zero-shot",
    type=str,
    help="Evaluation mode, choose from {zero-shot, 1-shot, 2-shot, 5-shot, 10-shot, \
        random-1-shot, random-2-shot, random-5-shot, random-10-shot}"
)
argparser.add_argument(
    "--method",
    default=None,
    type=str,
    help="Four methods for evaluation, choose from {1, 2, 3, 4}")
argparser.add_argument(
    "--data_path",
    default="/data/natural_questions/simplified/v1.0-simplified_nq-dev-all.jsonl.gz",
    type=str,
    help="Path to the Natural Questions dev set")
argparser.add_argument(
    "--dataset",
    default="nq",
    type=str,
    help="Name of the dataset")
argparser.add_argument(
    "--db_path",
    default="/data/save/tfidf/nq_train_db.tsv",
    type=str,
    help="Path to the db file for retrieval")
argparser.add_argument(
    "--tfidf_path",
    default='/data/save/tfidf/nq_train_db-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz',
    type=str,
    help="Path to the trained tfidf parameters")

args = argparser.parse_args()


model_path = args.model_path
mode = args.mode
method = args.method

print("Load QA data samples ...")
start_time = time.time()
if args.dataset == "nq":
    gzip_data_file = open(args.data_path, "rb")
    with GzipFile(fileobj=gzip_data_file) as f:
        data = []
        for line in f:
            json_example = json.loads(line)
            data.append(simplify_nq_example(json_example))
elif args.dataset == "webquestions":
    data = load_dataset("web_questions")
    data = convert_webquestions_to_nq_format(data["test"])
elif args.dataset == "triviaqa":
    data = load_dataset("trivia_qa", "unfiltered.nocontext")
    data = convert_triviaqa_to_nq_format(data["validation"])
else:
    raise NotImplementedError

print("Loading data cost:", time.time() - start_time)


print("Load the model and the tokenizer ...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
model = model.cuda()


if mode != "zero-shot":
    if "random" in mode:
        random_retriever = RandomRetriever(args.db_path)
        mode_sp = mode.split("-")
        topk = int(mode_sp[0])
        seed = int(mode_sp[-1])
        qas = []
        for idx, sample in tqdm(enumerate(data), total=len(data)):
            qa = random_retriever.get_KB(seed, topk=topk)
            if len(qa) < topk:
                print(f"This sample only have {len(qa)}-shots!")
            qas.append(qa)
    else:
        # load retriever
        tfidf_retriever = TfidfRetriever(args.db_path, args.tfidf_path)

        topk = int(mode.replace("-shot", ""))
        qas = []
        for idx, sample in tqdm(enumerate(data), total=len(data)):
            qa = tfidf_retriever.get_KB(sample["question_text"], topk=topk)
            if len(qa) < topk:
                print(f"This sample only have {len(qa)}-shots!")
            qas.append(qa)

"""
In total, there're four ways of prompting, denoted as "method" == "1"/"2"/"3"/"4".
Method "1" and "2" are using "Question"/"Answer" as the trigger words, while
method "3" and "4" are using "Q"/"A". The difference between "1" & "2", and,
"3" & "4" is whether there're space following the colon after the "Answer"/"A". 
This may result in different results on the zero/few-shot QA tasks, which may 
related to the tokenization process of the models.
"""


questions = []
generations = []
golds = []
count = -1
for idx, sample in tqdm(enumerate(data), total=len(data)):
    if args.dataset != "nq":
        assert "answers" in sample
        short_answers = sample["answers"]
    else:
        document_tokens = sample["document_text"].split(" ")

        short_answers = []
        for annotation in sample['annotations']:
            for ans in annotation['short_answers']:
                ans_tokens = document_tokens[ans["start_token"]:ans["end_token"]]
                short_answers.append(" ".join(ans_tokens))

    if len(short_answers) == 0:
        continue
    else:
        count += 1
    
    question_text = sample["question_text"] 
    if not question_text.endswith("?"):
        question_text = question_text + "?"
    
    if method == "1":
        q_trigger = "Question: "
        a_trigger = "Answer: "
    elif method == "2":
        q_trigger = "Question: "
        a_trigger = "Answer:"
    elif method == "3":
        q_trigger = "Q: "
        a_trigger = "A: "
    elif method == "4":
        q_trigger = "Q: "
        a_trigger = "A:"
    else:
        raise ValueError("Invalid method name!")

    # Prepare the inputs to the models
    if mode == "zero-shot":
        encoder_text = q_trigger + question_text + " "  + a_trigger + "<mask>" 
        decoder_text = "</s><s> " + q_trigger + question_text + " " + a_trigger

        inputs = tokenizer(encoder_text, return_tensors="pt")["input_ids"]
        decoder_inputs = tokenizer(decoder_text, add_special_tokens=False, return_tensors="pt")["input_ids"]
    else: # few shot setting
        qa = qas[idx]

        encoder_text = []
        decoder_text = []
        for item in qa:
            encoder_text.append(q_trigger + item[0] + " " + a_trigger + item[1])
            decoder_text.append(q_trigger + item[0] + " " + a_trigger + item[1])

        encoder_text.append(q_trigger + question_text + " " + a_trigger +"<mask>")


        joint = " \n\n " 
        encoder_text = joint.join(encoder_text)

        inputs = tokenizer.encode(encoder_text)
        inputs = inputs[-974:]
        decoder_inputs = [tokenizer.eos_token_id] + inputs[:-2]
        inputs = torch.LongTensor([inputs])
        decoder_inputs = torch.LongTensor([decoder_inputs])


    # Generate the answer with prompts
    eos_token_id = tokenizer.eos_token_id
    gen_length = 128 if "</ent_desc>" in encoder_text else 50
    seq = model.generate(inputs.cuda(), decoder_input_ids=decoder_inputs.cuda(), max_length=decoder_inputs.size(1) + gen_length, eos_token_id=eos_token_id) 
    gen_text = tokenizer.batch_decode(seq[:, decoder_inputs.size(1):], clean_up_tokenization_spaces=True) 

    # Post-process on the generated answers
    gen_text = postprocess_text(gen_text[0])
    questions.append(question_text)
    generations.append(gen_text.replace("</s>", "").strip())
    golds.append("<SEP>".join(short_answers))


# Save the generations into files
os.makedirs(args.save_path, exist_ok=True)
with open(f"{args.save_path}/{args.dataset}_questions.txt", "w") as f:
    for line in questions:
        f.write(line.replace("\n", " ") + "\n")

with open(f"{args.save_path}/{args.dataset}_generations.txt", "w") as f:
    for line in generations:
        f.write(line.replace("\n", " ") + "\n")

with open(f"{args.save_path}/{args.dataset}_answers.txt", "w") as f:
    for line in golds:
        f.write(line.replace("\n", " ") + "\n")


# Gather all the answers
answers = []
for a in golds:
    answers.append(a.strip().split("<SEP>"))


# Evaluate the results
metric_f1 = F1Evaluator()
print(metric_f1.compute(predictions=generations, references=answers, cover_em=True))

