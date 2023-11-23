#!/bin/bash

# Set the paths to your Python interpreter and your script
PYTHON_EXECUTABLE="python"  # Replace with the path to your Python executable
# PYTHON_EXECUTABLE="CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port 56667"
SCRIPT_PATH="train/nci.py"  # Replace with the actual path to your Python script

# Set the arguments for your Python script
# Adjust these values based on your requirements
TRAIN_CSV="/work/pi_hongyu_umass_edu/zhichao/GenRetrieval/ICD/data/mimic3/MIMIC_TRAIN_SEMANTIC.csv"
VAL_CSV="/work/pi_hongyu_umass_edu/zhichao/GenRetrieval/ICD/data/mimic3/MIMIC_DEV_SEMANTIC.csv"
TEST_CSV="/work/pi_hongyu_umass_edu/zhichao/GenRetrieval/ICD/data/mimic3/MIMIC_TEST_SEMANTIC.csv"
MODEL_TYPE="google/long-t5-tglobal-base"
TOKENIZER_TYPE="google/long-t5-tglobal-base"
ICD9_TREE_PATH="/work/pi_hongyu_umass_edu/zhichao/GenRetrieval/ICD/data/mimic3/mimic_train_tree.pkl"
MAX_LENGTH=4096
MAX_OUT_LENGTH=10
EOS_TOKEN=1
LABEL_COL="semantic_id"
SHARE_ENC_DEC_EMBEDDINGS=0
DECODE_VOCAB_SIZE=101
USE_PAWA_DECODER=1
PAWA_DECODER_HEADS=8
PAWA_NUM_LAYERS=4
TRAIN_BATCH_SIZE=2
EVAL_BATCH_SIZE=4
GRAD_ACCUM_STEPS=1
TRAIN_LOGGING_STEPS=10
EVAL_LOGGING_STEPS=500000
SAVE_STEPS=500000
LOGGING_FIRST_STEP=1
LR=2e-4
EPOCHS=10
ES_THRESHOLD=0.005
PATIENCE=3
SEED=0
EXP_NAME=long-t5-tglobal-base
BEAM_SIZE=15
METRIC_FOR_BEST_MODEL=eval_f1_at_15
FP16=0
FP16_OPT_LEVEL=O2


# Run the Python script with the specified arguments
$PYTHON_EXECUTABLE $SCRIPT_PATH \
    --train_csv $TRAIN_CSV \
    --val_csv $VAL_CSV \
    --test_csv $TEST_CSV \
    --model_type $MODEL_TYPE \
    --tokenizer_type $TOKENIZER_TYPE \
    --icd9_tree_path $ICD9_TREE_PATH \
    --max_length $MAX_LENGTH \
    --max_out_length $MAX_OUT_LENGTH \
    --eos_token $EOS_TOKEN \
    --label_col $LABEL_COL \
    --share_enc_dec_embeddings $SHARE_ENC_DEC_EMBEDDINGS \
    --decode_vocab_size $DECODE_VOCAB_SIZE \
    --use_pawa_decoder $USE_PAWA_DECODER \
    --pawa_decoder_heads $PAWA_DECODER_HEADS \
    --pawa_num_layers $PAWA_NUM_LAYERS \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --grad_accum_steps $GRAD_ACCUM_STEPS \
    --train_logging_steps $TRAIN_LOGGING_STEPS \
    --eval_logging_steps $EVAL_LOGGING_STEPS \
    --save_steps $SAVE_STEPS \
    --logging_first_step $LOGGING_FIRST_STEP \
    --lr $LR \
    --epochs $EPOCHS \
    --es_threshold $ES_THRESHOLD \
    --patience $PATIENCE \
    --seed $SEED \
    --exp_name $EXP_NAME \
    --beam_size $BEAM_SIZE \
    --metric_for_best_model $METRIC_FOR_BEST_MODEL \
    --fp16 $FP16 \
    --fp16_opt_level $FP16_OPT_LEVEL
