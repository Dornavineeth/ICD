import os
import sys
import torch
import pickle
import random
import argparse
import numpy as np
from collections import defaultdict
sys.path.append(".")

from transformers import (
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    GenerationConfig
)
from dataset.mimic3 import (
    get_mimic3_dataset, 
    logit_mask_fn,
    get_decode_vocab_size,
    decode_label
)
from dataset.icd9 import Node, ICD9
from models import get_model
from trie import Trie
from train.evaluation import (
    get_labels_mapping,
    get_y_y_hat_matrix,
    all_metrics
)


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser(description="MIMIC3 Training Script")    
    # model
    parser.add_argument("--model_type", type=str, default='google/long-t5-tglobal-base', help="Model type")
    parser.add_argument("--tokenizer_type", type=str, default='google/long-t5-tglobal-base', help="Tokenizer type")
    parser.add_argument("--share_enc_dec_embeddings",  type=int, default=0, help="Share encoder and decoder embeddings")
    parser.add_argument("--decode_vocab_size", type=int, default=101, help="Decoder vocabulary size")
    parser.add_argument("--use_pawa_decoder", type=int, default=1, help="Use PAWA decoder")
    parser.add_argument("--pawa_decoder_heads", type=int, default=8, help="PAWA decoder heads")
    parser.add_argument("--pawa_num_layers", type=int, default=4, help="PAWA decoder num layers")
    
    # data
    parser.add_argument("--train_csv", type=str, default="path/to/train.csv", help="Path to the training CSV file")
    parser.add_argument("--val_csv", type=str, default="path/to/val.csv", help="Path to the validation CSV file")
    parser.add_argument("--test_csv", type=str, default="path/to/test.csv", help="Path to the test CSV file")
    parser.add_argument("--icd9_tree_path", type=str, default='path/to/icd9_tree.pkl', help="Path to the ICD9 tree pickle file")
    parser.add_argument("--max_length", type=int, default=4096, help="Maximum input sequence length")
    parser.add_argument("--max_out_length", type=int, default=10, help="Maximum output sequence length")
    parser.add_argument("--eos_token", type=int, default=1, help="EOS token")
    parser.add_argument("--label_col", type=str, default="semantic_id", help="Column name for labels")
    parser.add_argument("--label_pad_token_id", type=int, default=-100, help="pad token for lables")
    
    # training args
    parser.add_argument('--exp_name', type=str, required=True, help="Name of experiments")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=4, help="Evaluation batch size")
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--train_logging_steps", type=int, default=100, help="Number of train steps to log")
    parser.add_argument("--eval_logging_steps", type=int, default=5000, help="Number of train steps to log")
    parser.add_argument("--save_steps", type=int, default=5000, help="Number of steps to to save a model")
    parser.add_argument("--logging_first_step", type=int, default=0, help="Whether to do evaluation first")
    parser.add_argument("--es_threshold", type=float, default=0.005, help="Early stopping threshold")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--save_total_limit", type=int, default=3, help="save_total_limit")
    parser.add_argument("--beam_size", type=int, default=15, help="Beam size used for generations")
    parser.add_argument("--metric_for_best_model", type=str, default='eval_f1_at_15', help="Metric to track for checkpointing")
    parser.add_argument("--fp16", type=int, default=0, help="Use Fp16 mode")
    parser.add_argument("--fp16_opt_level", type=str, default="O0", help="fp16 optimization level")
    
    parser.add_argument('--output', type=str, default='./output/')
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument("--local_rank", type=int, default=0)

    return parser.parse_args()

def main():
    args = parse_args()
    seed_everything(args.seed)

    splits = {
        "train": args.train_csv,
        "val": args.val_csv,
        "test": args.test_csv
    }

    with open(args.icd9_tree_path, 'rb') as f:
        icd9_tree = pickle.load(f)
        icd9_tree._init_max_children_level()

    logit_mask_function = logit_mask_fn(icd9_tree, args.eos_token)
    model_kwargs = {
        "share_enc_dec_embeddings": bool(args.share_enc_dec_embeddings),
        "decode_vocab_size": args.decode_vocab_size,
        "use_pawa_decoder": bool(args.use_pawa_decoder),
        "pawa_decoder_heads": args.pawa_decoder_heads,
        "logit_mask_fn": logit_mask_function,
        "max_out_length": args.max_out_length,
        "pawa_num_layers": args.pawa_num_layers,
        "decode_vocab_size": get_decode_vocab_size(icd9_tree)
    }

    model, tokenizer, gen_config, config = get_model(args.model_type, args.tokenizer_type, **model_kwargs)
    print(config)

    dataset_kwargs = {
        "icd9_tree": icd9_tree, 
        "label_col": args.label_col, 
        "eos_token": args.eos_token
    }
    dataset = get_mimic3_dataset(splits, tokenizer, mode='train', max_length=args.max_length, max_out_length=args.max_out_length,**dataset_kwargs)

    trie = Trie(sequences=dataset["train"]["labels"], start_token=config.decoder_start_token_id)
    # model.trie = trie

    # Data collator
    label_pad_token_id = args.label_pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        padding='longest',
    )

    # Define training args
    repository_id = os.path.join(
        f"{args.output}", f"{args.exp_name}")


    def _compute_metrics(tokenizer, icd9_tree, eos_token=1, num_return_sequences=15):
        def dec_2d(dec, size):
            res = []
            i = 0
            while i < len(dec):
                res.append(dec[i: i + size])
                i = i + size
            return res
        
        def metrics(eval_pred):
            input_ids, predictions, labels = eval_pred.inputs, eval_pred.predictions, eval_pred.label_ids
            input_ids = np.where(input_ids != -100, input_ids, tokenizer.pad_token_id)
            predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            import pdb;pdb.set_trace()
            query = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            labels = [decode_label(x, icd9_tree, eos_token) for x in labels]
            predictions = [decode_label(x,  icd9_tree, eos_token) for x in predictions]
            predictions = dec_2d(predictions, num_return_sequences)
            # Create a dictionary to store aggregated predictions and ground truths for each query
            query_data = defaultdict(lambda: {"predictions": set(), "ground_truths": set()})

            all_labels = set()
            # Group by query and aggregate predictions and ground truths
            for q, pred, gt in zip(query, predictions, labels):
                query_data[q]["predictions"].update(pred)
                query_data[q]["ground_truths"].update(gt)
                all_labels.update(pred)
                all_labels.update(gt)
            semantic2id, _ = get_labels_mapping(all_labels)
            gts = [query_data[q]["ground_truths"] for q in query_data]
            preds = [query_data[q]["predictions"] for q in query_data]
            y_matrix, y_hat_raw_matrix = get_y_y_hat_matrix(semantic2id, preds, gts)
            import pdb;pdb.set_trace()
            all_metric_values = all_metrics(y_matrix, y_hat_raw_matrix)
            return all_metric_values
        return metrics

    compute_metrics =  _compute_metrics(tokenizer, icd9_tree, args.eos_token, args.beam_size)

    # gen_config.prefix_allowed_tokens_fn = lambda batch_id, sent: trie.get(sent.tolist())
    # trie
    # model.generation_config = gen_config
    gen_config.num_return_sequences =  args.beam_size
    gen_config.early_stopping = True
    gen_config.num_beams = args.beam_size
    gen_config.max_length = args.max_out_length
    # import pdb;pdb.set_trace()
    # "prefix_allowed_tokens_fn" : lambda batch_id, sent: trie.get(sent.tolist()),
    training_args = Seq2SeqTrainingArguments(
        output_dir=repository_id,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,

        # # evaluation
        # predict_with_generate=True,
        # include_inputs_for_metrics=True,
        # metric_for_best_model=args.metric_for_best_model,
        # generation_config=gen_config,

        # logging & evaluation strategies
        logging_dir=f"{repository_id}/logs",
        logging_strategy="steps",
        logging_steps=args.train_logging_steps,
        evaluation_strategy="steps" if "val" in dataset.keys() else "no",
        eval_steps=args.eval_logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True if "val" in dataset.keys() else False,
        logging_first_step=bool(args.logging_first_step),

        # optimization
        fp16=bool(args.fp16),
        fp16_opt_level=args.fp16_opt_level,
        
        # push to hub parameters
        report_to="tensorboard",
        seed=args.seed,
    )

     # Create Trainer instance
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"] if "val" in dataset.keys() else None,
        callbacks=[EarlyStoppingCallback(early_stopping_threshold=args.es_threshold,
                                         early_stopping_patience=args.patience)] if "val" in dataset.keys() else None,
        # compute_metrics=compute_metrics,
    )
    trainer.train()

if __name__ == "__main__":
    main()
