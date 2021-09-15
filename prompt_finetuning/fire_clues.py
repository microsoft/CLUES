import os

models = ['bert-base-uncased', 'bert-large-uncased', 'roberta-large', 'microsoft/deberta-large']
#lrs = ['1e-4', '1e-5', '2e-6', '2e-6']
lrs = ['1e-5'] * 4
tasks = ['CLUE-SST-2', 'CLUE-MNLI']
#tasks = ['CLUE-SST-2']
#tasks = ['CLUE-MNLI']
CLUES_SEEDS = [1,2,3,4,5]
Ks = [10, 20, 30]

cnt = 0
for i, model in enumerate(models):
    for task in tasks:
        for clues_seed in CLUES_SEEDS:
            for K in Ks:
                if 'SST' in task:
                    data_dir = 'data/CLUES/SST-2'
                    data_train = 'sst_%d_%d.tsv' % (K, clues_seed)
                    data_test = 'dev.tsv'
                else:
                    data_dir = 'data/CLUES/MNLI'
                    data_train = 'mnli_%d_%d.txt' % (K, clues_seed)
                    data_test = 'dev.tsv'

                if K % 8 == 0:
                    max_step = str(20 * K // 8)
                    eval_step = str(K//8)
                else:
                    max_step = str(20 * (K // 8 + 1))
                    eval_step = str(K//8 + 1)

                gpu = (cnt % 4)
                cmd = 'CUDA_VISIBLE_DEVICES=%d MODEL=%s CLUES_SEED=%d TASK=%s DATA_DIR=%s DATA_TRAIN=%s DATA_TEST=%s LR=%s MAX_STEP=%s EVAL_STEP=%s REAL_BS=8 TYPE=prompt-demo sh run_clue_full.sh' % (gpu, model, clues_seed, task, data_dir, data_train, data_test, lrs[i], max_step, eval_step)
                cnt += 1
                print (cmd)

#cmd='CUDA_VISIBLE_DEVICES=0 MODEL=bert-base-uncased CLUES_SEED=1 TASK=CLUE-MNLI DATA_DIR=data/CLUES/MNLI DATA_TRAIN=mnli_full.txt DATA_TEST=dev.tsv LR=1e-5 MAX_STEP=50000 EVAL_STEP=100 REAL_BS=16 TYPE=prompt sh run_clue_full.sh            

#cmd='CUDA_VISIBLE_DEVICES=2 MODEL=roberta-base CLUES_SEED=1 TASK=CLUE-SST-2 DATA_DIR=data/CLUES/SST-2 DATA_TRAIN=sst_10_1.tsv DATA_TEST=dev.tsv sh run_clue.sh'
