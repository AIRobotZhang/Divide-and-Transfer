#!/bin/bash
GPUID=$1
echo "Run on GPU $GPUID"
TRAIN=$3
TEST=$4
# data
DATASET=$2
DATASET_SRC=$5
PROJECT_ROOT=$(dirname "$(readlink -f "$0")")
DATA_ROOT=$PROJECT_ROOT/dataset/

EVAL=$6
# model
TOKENIZER_TYPE=bert
SPAN_TYPE=bert
TYPE_TYPE=bert
TOKENIZER_NAME=bert-base-uncased
SPAN_MODEL_NAME=bert-base-uncased
TYPE_MODEL_NAME=bert-base-uncased

# params
LR=1e-5
WEIGHT_DECAY=1e-4
EPOCH=50
SEED=0

ADAM_EPS=1e-8
ADAM_BETA1=0.9
ADAM_BETA2=0.98
WARMUP=500

TRAIN_BATCH=32
TRAIN_BATCH_TGT=32
EVAL_BATCH=32
DELTA=$7

OUTPUT=$PROJECT_ROOT/ptms/$DATASET/

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPUID python3 -u run_script.py --data_dir $DATA_ROOT \
  --span_model_name_or_path $SPAN_MODEL_NAME \
  --type_model_name_or_path $TYPE_MODEL_NAME \
  --output_dir $OUTPUT \
  --tokenizer_name_or_path $TOKENIZER_NAME \
  --cache_dir $PROJECT_ROOT/cached_models \
  --max_seq_length 128 \
  --learning_rate $LR \
  --weight_decay $WEIGHT_DECAY \
  --adam_epsilon $ADAM_EPS \
  --adam_beta1 $ADAM_BETA1 \
  --adam_beta2 $ADAM_BETA2 \
  --max_grad_norm 1.0 \
  --num_train_epochs $EPOCH \
  --warmup_steps $WARMUP \
  --per_gpu_train_batch_size $TRAIN_BATCH \
  --per_gpu_train_batch_size_tgt $TRAIN_BATCH_TGT \
  --per_gpu_eval_batch_size $EVAL_BATCH \
  --gradient_accumulation_steps 1 \
  --logging_steps 100 \
  --save_steps 100000 \
  --do_train $TRAIN\
  --do_test $TEST \
  --evaluate_during_training \
  --seed $SEED \
  --overwrite_output_dir \
  --model_type $TOKENIZER_TYPE \
  --dataset $DATASET \
  --dataset_src $DATASET_SRC \
  --do_lower_case \
  --delta $DELTA \
  --eval $EVAL \
  # --continu \

