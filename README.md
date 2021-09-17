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

## How do I cite CLUES?

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
