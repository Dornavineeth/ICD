import torch
import pickle

import sys
sys.path.append('.')

from dataset.mimic3 import (
    get_mimic3_dataset,
    logit_mask_fn,
    decode_label
)
from trie import Trie
from dataset.icd9 import Node, ICD9
from models import get_model

splits= {
    "train": "/work/pi_hongyu_umass_edu/zhichao/GenRetrieval/Neural-Corpus-Indexer-NCI/Data_process/mimic3_small/MIMIC_TRAIN_SEMANTIC.csv",
    "val": "/work/pi_hongyu_umass_edu/zhichao/GenRetrieval/Neural-Corpus-Indexer-NCI/Data_process/mimic3_small/MIMIC_DEV_SEMANTIC.csv",
    "test": "/work/pi_hongyu_umass_edu/zhichao/GenRetrieval/Neural-Corpus-Indexer-NCI/Data_process/mimic3_small/MIMIC_VAL_SEMANTIC.csv"
}
model_type = 'google/long-t5-tglobal-base'
tokenizer_type = 'google/long-t5-tglobal-base'
icd9_tree_path = '/work/pi_hongyu_umass_edu/zhichao/GenRetrieval/Neural-Corpus-Indexer-NCI/Data_process/mimic3_small/mimic_train_tree.pkl'
max_length = 4096
max_out_length = 10
eos_token = 1
label_col = "semantic_id"
share_enc_dec_embeddings=False
decode_vocab_size = 101
use_pawa_decoder = True
pawa_decoder_heads = 8
pawa_num_layers = 4


with open(icd9_tree_path, 'rb') as f:
    icd9_tree = pickle.load(f)
    icd9_tree._init_max_children_level()


logit_mask_function = logit_mask_fn(icd9_tree, eos_token)
model_kwargs = {
    "share_enc_dec_embeddings":share_enc_dec_embeddings,
    "decode_vocab_size":decode_vocab_size,
    "use_pawa_decoder":use_pawa_decoder,
    "pawa_decoder_heads":pawa_decoder_heads,
    "logit_mask_fn":logit_mask_function,
    "max_out_length": max_out_length,
    "pawa_num_layers": pawa_num_layers
}

model, tokenizer, gen_config, config = get_model(model_type, tokenizer_type, **model_kwargs)
dataset = get_mimic3_dataset(splits, tokenizer, mode='train', max_length=max_length, max_out_length=10, **{"icd9_tree": icd9_tree, "label_col":label_col, "eos_token":eos_token})

input_ids = torch.tensor(dataset['train'][0]['input_ids']).view(1,-1)
attention_mask = torch.tensor([dataset['train'][0]['attention_mask']]).view(1,-1)
labels = torch.tensor([dataset['train'][0]['labels']]).view(1,-1)
trie = Trie(sequences=dataset["train"]["labels"], start_token=0)

output = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    labels=labels
    )

# def f(batch_id, sent):
#     res = trie.get(sent.tolist())
#     print(batch_id, sent, res)
#     return res

out_gen = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    max_length=10,
    num_return_sequences=10,
    num_beams=10,
    prefix_allowed_tokens_fn = lambda batch_id, sent: trie.get(sent.tolist())
    # prefix_allowed_tokens_fn=f,
)