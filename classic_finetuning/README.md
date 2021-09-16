[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


# Classic finetuning for CLUES


## Quickstart

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


## Notes and Acknowledgments
MT-DNN: https://github.com/namisan/mt-dnn <br/>

### How do I cite CLUES?

```

@article{cluesteam2021,
  title={Few-Shot Learning Evaluation in Natural Language Understanding},
  author={Mukherjee, Subhabrata and Liu, Xiaodong and Zheng, Guoqing and Hosseini, Saghar and Cheng, Hao and Yang, Greg and Meek, Christopher and Awadallah, Ahmed Hassan and Gao, Jianfeng},
  year={2021}
}

@inproceedings{liu2020mtdnn,
  title={The Microsoft Toolkit of Multi-Task Deep Neural Networks for Natural Language Understanding},
  author={Liu, Xiaodong and Wang, Yu and Ji, Jianshu and Cheng, Hao and Zhu, Xueyun and Awa, Emmanuel and He, Pengcheng and Chen, Weizhu and Poon, Hoifung and Cao, Guihong and others},
  booktitle={Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics: System Demonstrations},
  pages={118--126},
  year={2020}
}
```
