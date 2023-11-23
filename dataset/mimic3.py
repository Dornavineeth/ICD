import os
import torch
from datasets import load_dataset
from .utils import (
    clean_mimic3
)
from functools import partial
from .icd9 import ICD9, Node

def get_mimic3_dataset(splits, tokenizer, mode='train', max_length=4096, max_out_length=10, **kwargs):
    if mode == 'train':
        splits.pop('test')
    elif mode == 'test':
        splits = {"test":splits['test']}
    
    icd9_tree = kwargs.get("icd9_tree", None)
    eos_token = kwargs.get("eos_token", 1)
    label_col = kwargs.get("label_col", "semantic_id")

    dataset = load_dataset(
        'csv',
        data_files=splits,
        cache_dir=os.path.join("cache", "mimic3")
    )
    print("\n--- Data Info ---")
    for k in dataset.keys():
        print(f"{k} : {len(dataset[k])}")
    
    # preprocess dataset
    dataset = dataset.map(partial(clean, clean_func=clean_mimic3),
                          batched=True)
    
    # tokenizer inputs
    dataset = dataset.map(partial(tokenize, tokenizer=tokenizer),
                          batched=True)
    

    # encoder labels
    dataset = dataset.map(partial(encode_labels, icd9_tree=icd9_tree, label_col= label_col, eos_token=eos_token),
                          batched=True)
    
    greater_tokens_freq = len(dataset['train'])

    # filter the dataset
    dataset = dataset.filter(lambda example: len(example["input_ids"]) < max_length and len(example["labels"]) < max_out_length, num_proc=22)
    print("Examples having more than max_length input tokens : ", greater_tokens_freq -  len(dataset['train']))
    
    return dataset


def tokenize(examples, tokenizer, mode='train'):
    model_inputs = tokenizer(examples["query"])
    return {**model_inputs,**examples}


def encode_labels(examples, icd9_tree, label_col="semantic_id", eos_token=1):
    labels = []
    for i in range(len(examples[list(examples.keys())[0]])):
        label = examples[label_col][i]
        label = encode_single_label(label, icd9_tree, eos_token)
        labels.append(label)
    examples["labels"] = labels
    return examples

def decode_labels(seqs, icd9_tree, eos_token=1):
    results = []
    for seq in seqs:
        results.apped(decode_label(seq, icd9_tree, eos_token))
    return results

def get_decode_vocab_size(icd9_tree):
    decode_vocab_size =  sum([icd9_tree.max_children_level(l) for l in range(icd9_tree.subtree_depth() - 1)]) + 2
    return decode_vocab_size


def decode_label(seq, icd9_tree, eos_token=1, pad_token=0):
    '''
    Param:
        seqs: 2d ndarray to be decoded
    Return:
        doc_id string, List[str]
    '''
    try:
        eos_idx = seq.tolist().index(eos_token)
        if seq[0] == pad_token:
            seq = seq[1: eos_idx]
        else:
            seq = seq[0: eos_idx]
    except:
        print("no eos token found")
    res = []
    for i, s in enumerate(seq):
        offset =  2
        if i >  icd9_tree.subtree_depth()-1:
            print("Incorrect Sequence")
            return "-1"
        for l in range(i):
            off = icd9_tree.max_children_level(l)
            if off is None:
                print("Going beyond depth")
                break
            offset += off
        if s-offset>=0:
            res.append(s-offset)
        else:
            print(seq, s, offset)
            print("Invalid Input of seq")
    return '-'.join(str(c) for c in res)

def encode_single_label(seq, icd9_tree, eos_token=1):
    '''
    Param:
        seq: doc_id string to be encoded, like "23456"
    Return:
        List[Int]: encoded tokens
    '''
    target_id_int = []
    for i, c in enumerate(seq.split('-')):
        cur_token =  sum([icd9_tree.max_children_level(l) for l in range(i)]) + int(c) + 2
        target_id_int.append(cur_token)
    return torch.tensor(target_id_int + [eos_token])  # append eos_token

def clean(examples, clean_func):
    queries = []
    for i in range(len(examples[list(examples.keys())[0]])):
        query = examples["query"][i]
        query = clean_func(query)
        queries.append(query)
    examples["query"] = queries
    return examples

def logit_mask_fn(icd9_tree, eos_token=1):
    def f(t):
        if t>icd9_tree.subtree_depth()-1:
            return []
        elif t==icd9_tree.subtree_depth()-1:
            return [eos_token]
        offset = sum([icd9_tree.max_children_level(l) for l in range(t)]) + 2
        idx = [eos_token] + [x+offset for x in range(icd9_tree.max_children_level(t))]
        return idx
    return f


