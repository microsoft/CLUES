[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

# CLUES: Few-Shot Learning Evaluation in Natural Language Understanding

This repo contains the source code for baseline models in [CLUES](https://openreview.net/pdf?id=VhIIQBm00VI) under [MIT License](LICENSE).

## Overview

We release source codes for two fine-tuning strategies on CLUES, one with classic fine-tuning and the other with prompt-based fine-tuning.

## Classic finetuning

### Setup Environment
   1. ```> git clone git@github.com:microsoft/CLUES.git```
   1. ```> git clone git@github.com:namisan/mt-dnn.git```
   2. ```> cp -rf CLUES/classic_finetuning/ mt-dnn/ ```
   3. ```> cd mt-dnn/ ```

### Run Experiments
   1. Preprocess data </br>
      ```> bash run_clues_data_process.sh <CLUES-DATA> <CLUES-TASK-DEF: e.g., experiments/clues/clues_ext_task_def.yml>```


   2. Train/test Models </br>
      ```> bash run_clues_batch.sh <CLUES-DATA> <MODEL-SIZE: large or base>```

## Prompt fine-tuning

### Setup
   1. ```cd prompt_finetuning```
   2. Run ```sh setup.sh``` to automatically fetch dependency codebase and apply our patch for CLUES

### Run Experiments

   All prompt-based funetuning baselines run commands are in `experiments.sh`, simple run by `sh experiments.sh`

## Leaderboard

Here we maintain a leaderboard, allowing researchers to submit their results as entries.

### Submission Instructions

- Each submission must be submitted as a pull request modifying the markdown file underlying the leaderboard.
- The submission must attach an accompanying public paper and public source code for reproducing their results on our dataset. 
- A submission can be toward any subset of tasks in our benchmark, or toward the aggregate leaderboard. 
- For any task targeted by the submission, we require evaluation on (1) 10, 20, *and* 30 shots, and (2) all 5 splits of the corresponding dataset and a report of their mean and standard deviation.
- Each leaderboard will be sorted by the 30-shot mean accuracy.
- The submission should not use external data or data from other splits during few-shot finetuning, either as extra training set or as validation set for hyperparameter tuning.
- In the table entry, the submission should include a method name and a citation, hyperlinking to their publicly released source code reproducing the results. See the last entry of the table below for an example.

### Benchmarking CLUES for Aggregate 30-shot Evaluation

Abbreviations:
- FT = (classic) finetuning
- PT = prompt based tuning
- ICL = in-context learning, in the style of GPT-3
- μ+-σ = mean μ and standard deviation σ across our 5 splits


| Shots (K=30)     | SST-2       | MNLI        | CoNLL03     | WikiANN     | SQuAD-v2    | ReCoRD      | Average      |
|------------------|-------------|-------------|-------------|-------------|-------------|-------------|--------------|
| Human            | 83.7        | 69.4        | 87.4        | 82.6        | 73.5        | 91.9        | 81.4         |
| T5-Large-770M-FT | 52.3 +- 2.9 | 36.8 +- 3.8 | 51.2 +- 0.1 | 62.4 +- 0.6 | 43.7 +- 2.7 | 12 +- 3.8   | 43.1 +- 6.7  |
| BERT-Large-336M-FT    | 55.4 +- 2.5 | 33.3 +- 1.4 | 51.3 +- 0   | 62.5 +- 0.6 | 35.3 +- 6.4 | 14.9 +- 3.4 | 42.1 +- 7.8  |
| BERT-Base-110M-FT     | 53.6 +- 5.5 | 35.4 +- 3.2 | 51.3 +- 0   | 62.8 +- 0   | 32.6 +- 5.8 | 13.1 +- 3.3 | 41.5 +- 9.2  |
| DeBERTa-Large-400M-FT | 47.7 +- 9.0 | 26.7 +- 11  | 48.2 +- 2.9 | 58.3 +- 6.2 | 38.7 +- 7.4 | 21.1 +- 3.6 | 40.1 +- 17.8 |
| RoBERTa-Large-355M-FT | 53.2 +- 5.6 | 34.0 +- 1.1 | 44.7 +- 2.6 | 48.4 +- 6.7 | 43.5 +- 4.4 | 16 +- 2.8   | 40.0 +- 10.6 |
| RoBERTa-Large-355M-PT | 90.2 +- 1.8 | 61.6 +- 3.5 |             |             |             |             |              |
| DeBERTa-Large-400M-PT | 88.4 +- 3.3 | 62.9 +- 3.1 |             |             |             |             |              |
| BERT-Large-336M-PT    | 82.7 +- 4.1 | 45.3 +- 2.0 |             |             |             |             |              |
| GPT3-175B-ICL   | 91.0 +- 1.6 | 33.2 +- 0.2 |             |             |             |             |              |
| BERT-Base-110M-PT     | 79.4 +- 5.6 | 42.5 +- 3.2 |             |             |             |             |              |
| [Example (lastname et al.)](link2code) | 0+-0 | 0+-0 | 0+-0  | 0+-0  | 0+-0  | 0+-0  | 0+-0 | 

## Individual Task Performance over Multiple Shots

### SST-2

| Shots (K)        | 10         | 20         | 30        | All   |
|------------------|------------|------------|-----------|-------|
| RoBERTa-Large PT | 88.8+-3.9  | 89.0+-1.1  | 90.2+-1.8 | 93.8  |
| GPT-3 (175B) ICL | 85.9+-3.7  | 92.0+-0.7  | 91.0+-1.6 | -     |
| DeBERTa-Large PT | 83.4+-5.3  | 87.8+-3.5  | 88.4+-3.3 | 91.9  |
| **Human**        | 79.8       | 83.0       | 83.7      | -     |
| BERT-Base PT     | 63.9+-10.0 | 76.7+-6.6  | 79.4+-5.6 | 91.9  |
| BERT-Large PT    | 63.2+-11.3 | 78.2+-9.9  | 82.7+-4.1 | 91.0  |
| BERT-Large FT    | 46.3+-5.5  | 55.5+-3.4  | 55.4+-2.5 | 99.1  |
| BERT-Base FT     | 46.2+-5.6  | 54.0+-2.8  | 53.6+-5.5 | 98.1  |
| T5-Large FT      | 51.2+-1.8  | 53.4+-3.2  | 52.3+-2.9 | 97.6  |
| DeBERTa-Large FT | 43.0+-11.9 | 40.8+-22.6 | 47.7+-9.0 | 100.0 |
| RoBERTa-Large FT | 38.4+-21.7 | 52.3+-5.6  | 53.2+-5.6 | 98.6  |
| [Example (lastname et al.)](link2code) | 0+-0| 0+-0 | 0+-0 | -      |

### MNLI

| Shots (K)        | 10         | 20         | 30         | All  |
|------------------|------------|------------|------------|------|
| **Human**        | 78.1       | 78.6       | 69.4       | -    |
| RoBERTa-Large PT | 57.7+-3.6  | 58.6+-2.9  | 61.6+-3.5  | 87.1 |
| DeBERTa-Large PT | 44.5+-8.2  | 60.7+-5.3  | 62.9+-3.1  | 88.1 |
| BERT-Large PT    | 41.7+-1.0  | 43.7+-2.1  | 45.3+-2.0  | 81.9 |
| BERT-Base PT     | 40.4+-1.8  | 42.1+-4.4  | 42.5+-3.2  | 81.0 |
| T5-Large FT      | 39.8+-3.3  | 37.9+-4.3  | 36.8+-3.8  | 85.9 |
| BERT-Base FT     | 37.0+-5.2  | 35.2+-2.7  | 35.4+-3.2  | 81.6 |
| RoBERTa-Large FT | 34.3+-2.8  | 33.4+-0.9  | 34.0+-1.1  | 85.5 |
| BERT-Large FT    | 33.7+-0.4  | 28.2+-14.8 | 33.3+-1.4  | 80.9 |
| GPT-3 (175B) ICL | 33.5+-0.7  | 33.1+-0.3  | 33.2+-0.2  | -    |
| DeBERTa-Large FT | 27.4+-14.1 | 33.6+-2.5  | 26.7+-11.0 | 87.6 |

### CoNLL03

| Shots (K)        | 10        | 20        | 30        | All  |
|------------------|-----------|-----------|-----------|------|
| **Human**        | 87.7      | 89.7      | 87.4      | -    |
| BERT-Base FT     | 51.3+-0   | 51.3+-0   | 51.3+-0   | -    |
| BERT-Large FT    | 51.3+-0   | 51.3+-0   | 51.3+-0   | 89.3 |
| RoBERTa-Large FT | 50.8+-0.5 | 44.6+-5.1 | 44.7+-2.6 | 93.2 |
| DeBERTa-Large FT | 50.1+-1.2 | 47.8+-2.5 | 48.2+-2.9 | 93.6 |
| T5-Large FT      | 46.3+-6.9 | 50.0+-0.7 | 51.2+-0.1 | 92.2 |

### WikiANN

| Shots (K)        | 10        | 20        | 30        | All  |
|------------------|-----------|-----------|-----------|------|
| **Human**        | 81.4      | 83.5      | 82.6      | -    |
| BERT-Base FT     | 62.8+-0   | 62.8+-0   | 62.8+-0   | 88.8 |
| BERT-Large FT    | 62.8+-0   | 62.6+-0.4 | 62.5+-0.6 | 91.0 |
| T5-Large FT      | 61.7+-0.7 | 62.1+-0.2 | 62.4+-0.6 | 87.4 |
| DeBERTa-Large FT | 58.5+-3.3 | 57.9+-5.8 | 58.3+-6.2 | 91.1 |
| RoBERTa-Large FT | 58.5+-8.8 | 56.9+-3.4 | 48.4+-6.7 | 91.2 |

### SQuAD v2

| Shots (K)        | 10        | 20         | 30        | All  |
|------------------|-----------|------------|-----------|------|
| **Human**        | 71.9      | 76.4       | 73.5      | -    |
| BERT-Base FT     | 46.0+-2.4 | 34.9+-9.0  | 32.6+-5.8 | 76.3 |
| T5-Large FT      | 43.6+-3.5 | 28.7+-13.0 | 43.7+-2.7 | 87.2 |
| BERT-Large FT    | 42.3+-5.6 | 35.8+-9.7  | 35.3+-6.4 | 81.8 |
| DeBERTa-Large FT | 41.4+-7.3 | 44.4+-4.5  | 38.7+-7.4 | 90.0 |
| RoBERTa-Large FT | 38.1+-7.2 | 40.1+-6.4  | 43.5+-4.4 | 89.4 |

### ReCoRD

| Shots (K)        | 10        | 20        | 30        | All  |
|------------------|-----------|-----------|-----------|------|
| **Human**        | 94.1      | 94.2      | 91.9      | -    |
| DeBERTa-Large FT | 15.7+-5.0 | 16.8+-5.7 | 21.1+-3.6 | 80.7 |
| RoBERTa-Large FT | 12.0+-1.9 | 9.9+-6.2  | 16.0+-2.8 | 80.3 |
| T5-Large FT      | 11.9+-2.7 | 11.7+-1.5 | 12.0+-3.8 | 77.3 |
| BERT-Base FT     | 10.3+-1.8 | 11.7+-2.4 | 13.1+-3.3 | 54.4 |
| BERT-Large FT    | 9.9+-5.2  | 11.8+-4.9 | 14.9+-3.4 | 66.0 |

## How do I cite CLUES?

```
@article{cluesteam2021,
  title={Few-Shot Learning Evaluation in Natural Language Understanding},
  author={Mukherjee, Subhabrata and Liu, Xiaodong and Zheng, Guoqing and Hosseini, Saghar and Cheng, Hao and Yang, Greg and Meek, Christopher and Awadallah, Ahmed Hassan and Gao, Jianfeng},
  year={2021}
}
```

## Acknowledgments
MT-DNN: https://github.com/namisan/mt-dnn <br/>
LM-BFF: https://github.com/princeton-nlp/LM-BFF <br/>

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
