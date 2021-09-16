#!/usr/bin/env bash

set -e

############################### 
# Training script for CLUES.
# It uses 32G v100 for the experiments. For a smaller mem, please consider to increase #gpus.
# By CLUES team
############################### 

if [[ $# -lt 6 ]]; then
  echo $#
  echo "run_clues_finetune.sh <data_dir> <model_type> <model_size> <task> <num_gpu> <fewshot/full>"
  exit 1
fi

data_dir=$1
echo "Data dir: ${data_dir}"
model_type=$2
echo "Model type: ${model_type}"
model_size=$3
echo "Model size: ${model_size}"
# training set
task=$4
echo $task
num_gpus=$5
fewshot=$6
batch_size=${7:-"16"}


export ROOT_DIR="clues_mode"
export EPOCH=3
export LR="5e-5"
export OPTIM="adamax"
export MAX_ANSWER_LEN=20
export TASK_DEF="experiments/clues/clues_ext_task_def.yml"
export BS=${batch_size}

# For generative tasks.
#export TASK_DEF="experiments/clues/clues_gen_task_def.yml"

echo ${TASK_DEF}

train_dataset=${task}
test_dataset=${task}
echo "few shot: ${fewshot}"

if [ ${fewshot} == "fewshot" ]; then
  TASK=$(echo $task | cut -d'_' -f1)
  SHOT=$(echo $task | cut -d'_' -f2)
  ROUND=$(echo $task | cut -d'_' -f3)
  echo "training ${SHOT} of $TASK in ${ROUND}"
  test_dataset="${TASK}_fewshot"
else
  # full size trianing
  if [ ${task} == "mnli" ]; then
    test_dataset="mnli_matched,mnli_mismatched"
  else
    test_dataset=${task}
  fi
fi

echo "Training data: ${train_dataset}_train.json"
echo "Dev data: ${test_dataset}_dev.json"


if [ ${model_type} == "bert" ]; then
  MD="bert-${model_size}-uncased"
  DD="bert_${model_size}_uncased"
  ED=1
elif [ ${model_type} == "roberta" ]; then
  MD="roberta-${model_size}"
  DD="roberta_${model_size}_cased"
  ED=2
elif [ ${model_type} == "deberta" ]; then
  MD="microsoft/deberta-${model_size}"
  DD="deberta_${model_size}_cased"
  ED=6
elif [ ${model_type} == "t5e" ]; then
  MD="t5-${model_size}"
  DD="t5_${model_size}_cased"
  ED=8
elif [ ${model_type} == "t5g" ]; then
  MD="t5-${model_size}"
  DD="t5_${model_size}_cased_gen"
  ED=9
else
  echo "Unknown model ${model_type}"
  exit 1
fi


output_dir="${ROOT_DIR}/${task}/${DD}"
echo $output_dir
mkdir -p ${output_dir}

if [[ -f "${output_dir}/model*.pt" ]]; then
  rm "${output_dir}/model*.pt"
  rm "${output_dir}/config.json"
fi

echo "Training ${task} tokenized by ${DD} with ${MD}"

LOG_FILE="${output_dir}/mt-dnn-train.log"

if [ ${num_gpus} -ge 2 ]; then
  # multi gpu training
  # DDP config
  export MASTER_ADDR=localhost
  export MASTER_PORT="8787"
  export NNODES=1
  export NODE_RANK=0
  export GPUS_PER_NODE=${num_gpus}
  export WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
  export DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

  python -m torch.distributed.launch $DISTRIBUTED_ARGS train.py --data_dir=${data_dir}/${DD} --task_def=${TASK_DEF}  --train_dataset=${train_dataset} --test_dataset=${test_dataset} --init_checkpoint=${MD} --batch_size=${EPOCH} --learning_rate=${LR} --epochs=${EPS} --encoder_type=${ED} --optimizer=${OPTIM} --output_dir=${output_dir} --log_file=${LOG_FILE} --max_answer_len=${MAX_ANSWER_LEN}
else
  python train.py --data_dir=${data_dir}/${DD} --task_def=${TASK_DEF} --train_dataset=${train_dataset} --test_dataset=${test_dataset} --init_checkpoint=${MD} --batch_size=${BS} --learning_rate=${LR} --epochs=${EPOCH} --encoder_type=${ED} --optimizer=${OPTIM} --output_dir=${output_dir} --log_file=${LOG_FILE} --max_answer_len=${MAX_ANSWER_LEN}
fi

