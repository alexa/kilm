import os
import re
import json
import random
import argparse
from tqdm import tqdm
from collections import defaultdict
from functools import partial
from multiprocessing import Pool, cpu_count

unwanted_content = ['thumb', 'right', 'thumbnail', 'upright', 'left', 'alt',
                    'link=', 'bot=PearBOT 5', 'bot = PearBOT 5', 'Figure 1.',
                    'Fig.', 'A', 'An', '"',
                    'First edition (publ.', 'First edition  (publ.', 'frame',
                    'center', 'centre', 'none', 'top_scorer     =', '100x100px',
                    '220x124px', 'The', 'St.', 'St', 'Sir', '300x300px', '320x320px',
                    '350x350px', '400x400px', '250x250px', '200x200px', '150x150px',
                    '100000x260px', 'First edition', '}', 'border', 'frameless', 'Dr.',
                    'Entrance', 'Oblique', 'framed', 'First', 'Category puzzle']


def get_first_para(example, sep="=="):
    """
    Get the first paragraph of a Wikipedia article
    """
    data = json.loads(example)
    para_sample = {
        "title": data["title"],
        "short_description": data["short_description"],
        "first_sentence": data["first_sentence"]
    }

    text = data["text"]
    para_sample["text"] = text.split(sep)[0].strip()

    return para_sample


def parse_para(example):
    data = json.loads(example)
    para_sample = {
        "title": data["title"],
        "short_description": data["short_description"],
        "first_sentence": data["first_sentence"],
        "text": data["text"]
    }

    return para_sample


def filter_desc(example, record):
    """
    Filter out noisy short descriptions with hard-coded rules.
    """
    for k, v in example.items():
        if "may refer to:" in v:
            return None
        if v.lower().startswith("redirect"):
            return None
        if len(v) == 0:
            return None
        if k.startswith('Portal:') or k.startswith('MediaWiki:'):
            return None
        if "|" in v:
            line_sp = v.split("|")
            clean_line = []
            for sp in line_sp:
                if not sp.startswith("http:") and \
                        not sp.startswith("https:") and \
                        not sp.startswith("File:") and \
                        not (sp in k):
                    if "=" in sp:
                        continue
                    if sp.strip() in unwanted_content:
                        continue
                    if sp.endswith("px") and len(sp[:-2].split('x')[0].strip()) < 5:
                        continue
                    clean_line.append(sp)

            c = [record[l] for l in clean_line]
            if len(clean_line) == 0:
                return None
            elif clean_line == ['use both this parameter and ',
                                "birth_date to display the person's date of birth, "
                                "date of death, and age at death) -- >"]:
                return None
            elif clean_line == ['Portrait'] or \
                    clean_line == ['David'] or \
                    clean_line == ['William'] or \
                    clean_line == ['Charles'] or \
                    clean_line == ['James'] or \
                    clean_line == ['George'] or \
                    clean_line == ['description']:
                return None
            elif clean_line == ['300']:
                return None
            else:
                if len(c) > 1:
                    index = c.index(min(c))
                    return {k: clean_line[index]}
                else:
                    return {k: clean_line[0]}

        else:
            if "thumb" == v.strip():
                return None
            elif "right" == v.strip():
                return None
            elif "wikipedia list article" == v.lower().strip():
                return None
            else:
                return example


def filter_para(text):
    """
    Filter out noisy paragraphs with hard-coded rules.
    """
    # if the article content is missing
    if len(text) == 0:
        return True

    # if there's redirect information
    if text.lower().startswith("redirect"):
        return True

    # if the article is not well-cleaned-up
    if "File:" in text or "http:" in text or "https:" in text or "Image:" in text:
        return True
    # if there are too many \n in the article (which may because it doesn't have much content)
    new_line_rate = (len(text) - len(text.replace("\n", ""))) / len(text)
    if new_line_rate >= 0.01:
        return True
    # if there's any special case
    if len(re.findall(r"\[\[.*?in aviation.*?\]\]", text)) > 0:
        return True

    return False


def filter_doc(text):
    """
    Filter out noisy documents with hard-coded rules.
    """
    # if the article content is missing
    if len(text) == 0:
        return True

    # if there's redirect information
    if text.lower().startswith("redirect"):
        return True

    # if the article is not well-cleaned-up
    if "File:" in text or "http:" in text or "https:" in text or "Image:" in text:
        return True

    # if there's any special case
    if len(re.findall(r"\[\[.*?in aviation.*?\]\]", text)) > 0:
        return True
    return False


def process_entities(example, short_desc_dict=None, add_desc=True):
    """
    Remove missing entities and additional cleanup
    """
    assert short_desc_dict is not None
    text = example["text"]
    # sometimes, either `ent` or `content` is missing
    text = text.replace("[[]]", "")

    # match with pattern ``[[.*?]]'' to collect all the entities
    entities = re.findall(r"\[\[.*?\]\]", text)
    descs = []

    for item in entities:
        split_item = item[2:-2].split("|")
        if len(split_item) > 2:
            # skip the hyperlinks that are hard to parse
            continue

        # ent -> the wikipedia title for entity linking; content -> the actual content in wikipedia
        ent, content = split_item[0], split_item[-1]
        if len(ent) < 1 or len(content) < 1:
            text = text.replace(item, content)
            continue
        
        ent = ent.split("#")[0] if len(ent.split("#")[0]) != 0 else ent
        ent = ent[0].upper() + ent[1:]
        if ent not in short_desc_dict:
            text = text.replace(item, content)
        else:
            ent_desc = f"{item}<sep>{ent}<sep>{short_desc_dict[ent]}"
            descs.append(ent_desc)

    example["text"] = text
    if add_desc:
        example["desc"] = "<sep_ent>".join(descs)
    return example


def filter_samples(wiki_para_samples, desc_samples):
    """
    Filter out noisy samples for both paragraphs and the description samples.
    """

    # Filter out the data samples with "may refer to:", which takes around 3%
    desc_samples_filter = []
    wiki_para_samples_filter = []
    for para_sample in tqdm(wiki_para_samples, total=len(wiki_para_samples), desc="Filter out some para samples"):
        if not filter_para(para_sample["text"]):
            wiki_para_samples_filter.append(para_sample)
    
    # count entities
    record = defaultdict(int)
    for desc_line in tqdm(desc_samples, total=len(desc_samples)):
        desc_sample = json.loads(desc_line)
        for k, v in desc_sample.items():
            if "|" in v:
                for l in v.split("|"):
                    record[l] += 1

    for desc_line in tqdm(desc_samples, total=len(desc_samples), desc="Filter out some desc samples"):
        desc_sample = json.loads(desc_line)
        cleaned_sample = filter_desc(desc_sample, record)
        if cleaned_sample is not None:
            desc_samples_filter.append(cleaned_sample)

    # Remove the entities where the short description is missing
    desc_dict = {}
    for l in desc_samples_filter:
        desc_dict.update(l)

    clean_wiki_para_samples = []
    for sample in tqdm(
            wiki_para_samples_filter,
            total=len(wiki_para_samples_filter),
            desc="Remove missing entities & Collecting short descriptions"):
        clean_wiki_para_samples.append(process_entities(sample, short_desc_dict=desc_dict))

    return clean_wiki_para_samples, desc_samples_filter


def proc_data_samples(examples, desc_samples, sep="==", threads=1, tqdm_enabled=True):
    """
    Process data samples for the wikipedia summaries
    """
    threads = min(threads, cpu_count())
    with Pool(threads) as p:
        annotate_ = partial(
            get_first_para,
            sep=sep,
            )
        wiki_para_samples = list(
            tqdm(
                p.imap(annotate_, examples, chunksize=32),
                total=len(examples),
                desc="Get first paragraph out of wikipedia articles",
                disable=not tqdm_enabled,
            )
        )

    return filter_samples(wiki_para_samples, desc_samples)


def proc_data_samples_doc(examples, desc_samples, threads=1, tqdm_enabled=True):
    """
    Process data samples for the entire Wikipedia
    """
    threads = min(threads, cpu_count())
    with Pool(threads) as p:
        annotate_ = partial(
            parse_para,
            )
        wiki_para_samples = list(
            tqdm(
                p.imap(annotate_, examples, chunksize=32),
                total=len(examples),
                desc="Parse wikipedia articles",
                disable=not tqdm_enabled,
            )
        )

    return filter_samples(wiki_para_samples, desc_samples)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--data_folder",
        default="/dev/shm/data/wiki",
        type=str,
        help="The path to the folder for data")
    argparser.add_argument(
        "--threads",
        default=32,
        type=int,
        help="The path to the folder for data")
    argparser.add_argument(
        "--val_samples",
        default=1000,
        type=int,
        help="Number of samples in the validation set")
    argparser.add_argument(
        "--task",
        default="filter",
        type=str,
        help="The preprocessing task {filter, split}")
    args = argparser.parse_args()

    if args.task == "filter":
        with open(os.path.join(args.data_folder, "articles.json"), "r") as f:
            wike_lines = f.readlines()
        
        with open(os.path.join(args.data_folder, "short_descriptions.json"), "r") as f:
            desc_lines = f.readlines()
        
        wiki_para_samples, desc_samples = proc_data_samples(
            wike_lines, desc_lines, sep="==", threads=args.threads, tqdm_enabled=True
        )

        out_records = open(os.path.join(args.data_folder, "first_paragraphs_filter.json"), "w")
        out_desc_records = open(os.path.join(args.data_folder, "short_descriptions_filter.json"), "w")

        for record in wiki_para_samples:
            json.dump(record, out_records)
            out_records.write('\n')
        
        for record in desc_samples:
            json.dump(record, out_desc_records)
            out_desc_records.write('\n')

        out_records.close()
        out_desc_records.close()
    elif args.task == "filter:doc":
        with open(os.path.join(args.data_folder, "articles.json"), "r") as f:
            wike_lines = f.readlines()
        
        with open(os.path.join(args.data_folder, "short_descriptions_filter.json"), "r") as f:
            desc_lines = f.readlines()
        
        wiki_para_samples, desc_samples = proc_data_samples_doc(
            wike_lines, desc_lines, threads=args.threads, tqdm_enabled=True
        )

        out_records = open(os.path.join(args.data_folder, "paragraphs_filter.json"), "w")

        for record in wiki_para_samples:
            json.dump(record, out_records)
            out_records.write('\n')
        
        out_records.close()

    elif args.task == "split":
        random.seed(0)
        with open(os.path.join(args.data_folder, "first_paragraphs_filter.json"), "r") as f:
            wike_lines = f.readlines()
        
        random.shuffle(wike_lines)
        train_split = wike_lines[args.val_samples:]
        val_unseen_split = wike_lines[:args.val_samples]
        val_seen_split = train_split[:args.val_samples]

        with open(os.path.join(args.data_folder, "first_paragraphs_filter_train.json"), "w") as f:
            for line in train_split:
                f.write(line)
        
        with open(os.path.join(args.data_folder, "first_paragraphs_filter_valid_seen.json"), "w") as f:
            for line in val_seen_split:
                f.write(line)
        
        with open(os.path.join(args.data_folder, "first_paragraphs_filter_valid_unseen.json"), "w") as f:
            for line in val_unseen_split:
                f.write(line)

    elif args.task == "split:doc":
        random.seed(0)
        with open(os.path.join(args.data_folder, "paragraphs_filter.json"), "r") as f:
            wike_lines = f.readlines()
        
        random.shuffle(wike_lines)
        train_split = wike_lines[args.val_samples:]
        val_unseen_split = wike_lines[:args.val_samples]
        val_seen_split = train_split[:args.val_samples]

        with open(os.path.join(args.data_folder, "paragraphs_filter_train.json"), "w") as f:
            for line in train_split:
                f.write(line)
        
        with open(os.path.join(args.data_folder, "paragraphs_filter_valid_seen.json"), "w") as f:
            for line in val_seen_split:
                f.write(line)
        
        with open(os.path.join(args.data_folder, "paragraphs_filter_valid_unseen.json"), "w") as f:
            for line in val_unseen_split:
                f.write(line)
    else:
        raise NotImplementedError("The `task` argument should get one of `filter`, `filter:doc`, `split`, `split:doc`")
