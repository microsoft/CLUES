
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

export CLUES_DATA_PATH="${data_path}/mtdnn"
export CLUES_EXT_TASK_DEF=${clues_task_def}
mkdir -p ${CLUES_DATA_PATH}
# copy data to a shared folder


declare -a TASKS=('CoNLL2003' 'MNLI' 'ReCoRD' 'SQuAD-v2' 'SST-2' 'WikiANN_EN')
declare -a SHOTS=('10' '20' '30')
declare -a ROUNDS=('1' '2' '3' '4' '5' )

# copy data into a shared folder for mtdnn
for shot in "${SHOTS[@]}"
do
  for round in "${ROUNDS[@]}"
  do
    # cp CoNLL
    src="${data_path}/CoNLL2003/conll_train_${shot}_${round}.jsonl"
    tgt="${CLUES_DATA_PATH}/ner_${shot}_${round}_train.json"
    cp ${src} ${tgt}
    src="${data_path}/CoNLL2003/conll_test.jsonl"
    tgt="${CLUES_DATA_PATH}/ner_fewshot_dev.json"
    cp ${src} ${tgt}

    # cp MNLI
    src="${data_path}/MNLI/mnli_train_${shot}_${round}.jsonl"
    tgt="${CLUES_DATA_PATH}/mnli_${shot}_${round}_train.json"
    cp ${src} ${tgt}
    src="${data_path}/MNLI/mnli_test.jsonl"
    tgt="${CLUES_DATA_PATH}/mnli_fewshot_dev.json"
    cp ${src} ${tgt}

    # cp SST-2
    src="${data_path}/SST-2/sst_train_${shot}_${round}.jsonl"
    tgt="${CLUES_DATA_PATH}/sst_${shot}_${round}_train.json"
    cp ${src} ${tgt}
    src="${data_path}/SST-2/sst_test.jsonl"
    tgt="${CLUES_DATA_PATH}/sst_fewshot_dev.json"
    cp ${src} ${tgt}
    
    # cp ReCoRD
    src="${data_path}/ReCoRD/record_train_${shot}_${round}.jsonl"
    tgt="${CLUES_DATA_PATH}/record_${shot}_${round}_train.json"
    cp ${src} ${tgt}
    src="${data_path}/ReCoRD/record_test.jsonl"
    tgt="${CLUES_DATA_PATH}/record_fewshot_dev.json"
    cp ${src} ${tgt}

    # cp SQuAD-v2
    src="${data_path}/SQuAD-v2/squad-v2_train_${shot}_${round}.jsonl"
    tgt="${CLUES_DATA_PATH}/squad-v2_${shot}_${round}_train.json"
    cp ${src} ${tgt}

    src="${data_path}/SQuAD-v2/squad-v2_test.jsonl"
    tgt="${CLUES_DATA_PATH}/squad-v2_fewshot_dev.json"
    cp ${src} ${tgt}

    # cp WikiANN
    src="${data_path}/WikiANN_EN/wikiann_train_${shot}_${round}.jsonl"
    tgt="${CLUES_DATA_PATH}/wikiann_${shot}_${round}_train.json"
    cp ${src} ${tgt}

    src="${data_path}/WikiANN_EN/wikiann_test.jsonl"
    tgt="${CLUES_DATA_PATH}/wikiann_fewshot_dev.json"
    cp ${src} ${tgt}
  done
done


declare -a PLMS=('bert-base-uncased' 'bert-large-uncased' 'roberta-large' 't5-large' 'microsoft/deberta-base')

# preprocessing extractive clues data for mt-dnn
for plm in "${PLMS[@]}"
do
  python prepro_clues_ext.py --task_def ${CLUES_EXT_TASK_DEF} --root_dir ${CLUES_DATA_PATH} --model ${plm} 
done


# preprocessing generative clues data for mt-dnn
python prepro_clues_gen.py --task_def ${CLUES_EXT_TASK_DEF} --root_dir ${CLUES_DATA_PATH} --model t5-large