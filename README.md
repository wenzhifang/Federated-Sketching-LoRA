## Federated Sketching LoRA: On-Device Collaborative Fine-Tuning of Large Language Models

[![python](https://img.shields.io/badge/Python_3.10-306998?logo=python&logoColor=FFD43B)](https://www.python.org/downloads/release/python-31012/)
[![License: MIT](https://img.shields.io/badge/license-MIT-750014.svg)](https://opensource.org/licenses/MIT) 


## 🔥 Our Framework

TL, DR: In this repo, we provide the implementation of **Federated Sketching LoRA** (FSLoRA), a novel methodology for collaborative LLM fine-tuning across resource-heterogeneous devices

<div align="center">
    <img src="figures/Overview.png" alt="overview" style="width:60%;"/>
</div>


## 🖥️ Prerequisites

Install the required packages via:
```bash
pip install -r requirements.txt
```

Alternatively, ensure the following dependencies are installed:
```plaintext
python == 3.10.14
torch == 2.6.0
transformers == 4.50.1
peft == 0.14.0
accelerate == 1.2.1
bitsandbytes == 0.45.0
datasets == 3.2.0
vllm == 0.8.2
```

## 🗂️ Folder Structure
```
FSLoRA/
│   README.md
│   requirements.txt
│
├─── GLUE/
│   │   arg.py
│   │   main_slora_glue_het.py
│   │   models.py
│   │   data_utils.py
│   │   train_utils.py
│   │   run_main.sh
│
├─── Commensen_reasoning/
│   │   arg.py
│   │   LoRA_sketching_llama_het.py
│   │   evaluation_par.py
│   │   evaluation_vllm.py
│   │   models.py
│   │   utils_data.py
│   │   utils_data_wang.py
│   │   utils_train.py
│   │   main.py
│   │   run_main.sh
```
- **`GLUE/`**: Contains the primary codebase for RoBERTa on the GLUE benchmark.
  - `main_slora_glue_het.py`: Our FSLoRA framework for GLUE.
  - `models.py`: Includes building model.
  - `utils_data.py`: Data preprocessing.
  - `utils_data_wang.py`: Data preprocessing for Dolly, Apalca, and Widz.
  - `run_main.sh`: Passing args and call `main_slora_glue_het.py`.

- **`Commensen_reasoning/`**: Contains the primary codebase for LLaMA-3.2-3B on the Commensense Reasoning benchmark.
  - `LoRA_sketching_llama_het.py`: Our FSLoRA framework for LLaMA-3.2-3B.
  - `models.py`: Includes building model.
  - `evaluation_par.py`: For evaluation.
  - `run_main.sh`: Execute FSLoRA algorithm and evaluate the checkpoints

## Dataset
For the commonsense reasoning benchmark, data can be found at [Commonsense Reasoning Benchmark Dataset](https://github.com/AGI-Edgerunners/LLM-Adapters/tree/main/dataset)

## 🏃‍♂ Run Code

Run our framework with the following command:
```bash
python main/run_main.sh
```
This code runs with 4 NVIDIA A100 GPUs in parallel, using the Accelerate library for efficient multi-GPU support.
