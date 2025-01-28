## Federated Sketching LoRA: On-Device Collaborative Fine-Tuning of Large Language Models

[![python](https://img.shields.io/badge/Python_3.10-306998?logo=python&logoColor=FFD43B)](https://www.python.org/downloads/release/python-31012/)
[![License: MIT](https://img.shields.io/badge/license-MIT-750014.svg)](https://opensource.org/licenses/MIT) 

---
ICML Submission 2308

---

## 🔥 Our Framework

We present **Federated Sketching LoRA** (FSLoRA), a novel methodology that retains LoRA's scalability while adapting to the communication and computational capabilities of individual devices.

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
torch == 2.5.1
transformers == 4.47.1
peft == 0.14.0
accelerate == 1.2.1
bitsandbytes == 0.45.0
datasets == 3.2.0
```

## 🗂️ Folder Structure
```
FSLoRA/
│   README.md
│   requirements.txt
│
├─── Commensen_reasoning/
│   │   arg.py
│   │   LoRA_sketching_llama_het.py
│   │   evaluation_par.py
│   │   models.py
│   │   utils_data.py
│   │   utils_train.py
│   │   main.py
│   │   run_main.sh
│   │
├─── GLUE/
│   │   arg.py
```

- **`Commensen_reasoning/`**: Contains the primary codebase for the LLaMA-3.2-3B for Commensense Reasoning task.
  - `LoRA_sketching_llama_het.py`: Our FSLoRA framework for LLaMA-3.2-3B.
  - `models.py`: Includes building model.
  - `evaluation_par.py`: For evaluation.
  - `run_main.sh`: execute FSLoRA algorithm and evaluate the checkpoints


## 🏃‍♂ Run Code

Run our framework with the following command:
```bash
python main/run_main.sh
```
This code runs with 4 NVIDIA A100 GPUs in parallel, using the Accelerate library for efficient multi-GPU support.
