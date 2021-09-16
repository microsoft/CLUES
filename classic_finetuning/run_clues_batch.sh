#!/usr/bin/env bash


############################### 
# Batch training script for CLUES.
# By CLUES team
############################### 


if [[ $# -lt 2 ]]; then
  echo "run_clues_finetune.sh <data_dir> <model_size:large/base>"
  exit 1
fi

data_dir=$1
model_size=$2
export CLUES_DATA=${data_dir}

declare -a PLMS=('bert' 'roberta' 'deberta' 't5e' 't5g')
declare -a TASKS=('sst' 'mnli' 'squad-v2' 'record' 'ner' 'wikiann')
declare -a SHOTS=('10' '20' '30')
declare -a ROUNDS=('0' '1' '2' "3" "4")

## RUN fewshot experiments
for plm in "${PLMS[@]}"
do
   echo "$plm"
   if [ ${model_size} == "base" ]; then
     export NUM_GPUS=1
   else
     export NUM_GPUS=2
   fi
   
   for task in "${TASKS[@]}"
   do
     for shot in "${SHOTS[@]}"
     do
       for round in "${ROUNDS[@]}"
       do
         run_task="${task}_${shot}_${round}"
         echo "bash run_clues_finetune.sh ${CLUES_DATA} ${plm} ${model_size} ${run_task} ${NUM_GPUS} fewshot"
       done
     done
   done

done


## RUN full size experiments

for plm in "${PLMS[@]}"
do
   echo "$plm"
   if [ ${model_size} == "base" ]; then
     export NUM_GPUS=1
   else
     export NUM_GPUS=2
   fi
   
   for task in "${TASKS[@]}"
   do
    echo "bash run_clues_finetune.sh ${CLUES_DATA} ${plm} ${model_size} ${task} ${NUM_GPUS} full"
   done
done



