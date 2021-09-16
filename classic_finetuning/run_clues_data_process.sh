
#!/usr/bin/env bash

############################### 
# Data preprocess script for CLUES.
# By CLUES team
############################### 


if [[ $# -lt 2 ]]; then
  echo $#
  echo "run_clues_data_process.sh <data_dir> <task_def>"
  exit 1
fi

# export CLUES_DATA_PATH="../data/clues-data"
# export CLUES_EXT_TASK_DEF="experiments/clues/clues_ext_task_def.yml"
data_path=$1
clues_task_def=$2

export CLUES_DATA_PATH=${data_path}
export CLUES_EXT_TASK_DEF=${clues_task_def}

declare -a PLMS=('bert-base-uncased' 'bert-large-uncased' 'roberta-large' 't5-large' 'microsoft/deberta-base')

# preprocessing extractive clues data for mt-dnn
for plm in "${PLMS[@]}"
do
  python prepro_clues_ext.py --task_def ${CLUES_EXT_TASK_DEF} --root_dir ${CLUES_DATA_PATH} --model ${plm} 
done


# preprocessing generative clues data for mt-dnn
python prepro_clues_gen.py --task_def ${CLUES_EXT_TASK_DEF} --root_dir ${CLUES_DATA_PATH} --model t5-large