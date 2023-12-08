# LLM-for-InlineG
This repository stores the details of the paper "Multi-Intent Inline Code Comment Generation via Large Language Model". Here are some details about the project:

## Dataset

- This folder contains the dataset we used for our experiment

## Codebert

- This folder contains the Codebert model.
- The `model` folder contains the output results for three types of inline comments and code snippet pairs

## RQ1_generate_comment

- This folder contains the code of RQ1 in the paper "Multi-Intent Inline Code Comment Generation via Large Language Model", which is used in multi-intent inline comments generation based on LLM.
- The folder `concise_result` stores the output results.

## RQ1_random_shot

- This folder contains the code of RQ1 used in multi-intent inline comments generation based on LLM, where folder `result` stores the output results.

## RQ2_similar_shot

- This folder contains the code of RQ2 used in inline comments generation based on LLM, where folder `similar_result` stores the output results.

## RQ3_re_rank

- This folder contains the code of RQ3 used in inline comments generation based on LLM.

### Dependency
- pip install torch
- pip install transformers

### Prediction
* The model prediction is conducted on a machine with Nvidia GTX 1080 GPU, Intel(R) Core(TM) i7-6700 CPU and 16 GB RAM. The operating system is Ubuntu.
* Please refer to the paper for the detailed parameters of the model.
