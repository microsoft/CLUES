import os

models = ['bert-base-uncased', 'bert-large-uncased', 'roberta-large', 'microsoft/deberta-large']
lrs = ['1e-5'] * 4
tasks = ['CLUE-SST-2', 'CLUE-MNLI']
CLUES_SEEDS = [1,2,3,4,5]
Ks = [10, 20, 30]

cnt = 0
for i, model in enumerate(models):
    for task in tasks:
        for clues_seed in CLUES_SEEDS:
            for K in Ks:
                if 'SST' in task:
                    data_dir = '../data/CLUES/SST-2'
                    data_train = 'sst_train_%d_%d.jsonl' % (K, clues_seed)
                    data_test = 'sst_test.jsonl'
                else:
                    data_dir = '../data/CLUES/MNLI'
                    data_train = 'mnli_train_%d_%d.jsonl' % (K, clues_seed)
                    data_test = 'mnli_test.jsonl'

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
