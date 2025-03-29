## SimPER: A Minimalist Approach to Preference Alignment without Hyperparameters (ICLR 2025)

**TL; DR:** We propose a simpler yet more effective preference fine-tuning algorithm than DPO and SimPO without both hyperparameters and reference models. SimPER consistently and significantly  outperforms DPO and latest SimPO across   various settings.

### News 🎉
Our SimPER has been used as the training algorithm in the recent work [EXAONE Deep: Reasoning Enhanced Language Models](https://arxiv.org/pdf/2503.12524) from [LG AI Research](https://www.lgresearch.ai/). Their open models ([available on HuggingFace](https://huggingface.co/LGAI-EXAONE)) achieved better and competitive performance in math, science, and coding tasks compared to SOTA LLMs QwQ 32B and DeepSeek R1 671B!

### Environment

First, install PyTorch `2.1.2` from the [PyTorch Installation Page](https://pytorch.org/get-started/locally/).

Create a Python virtual environment using e.g. Conda:

```shell
conda create -n SimPER python=3.10 && conda activate SimPER
```

We provide an [environment file](https://github.com/tengxiao1/SimPER/blob/main/environment.yml) including the python package versions in our experiments for reproducibility. 

You will also need Flash Attention 2 installed, which can be done by running:

```shell
python -m pip install flash-attn --no-build-isolation
```

## Training Scripts

We provide four training config files for the four training setups reported in our paper. The training config is set for 4xA100 GPUs. 

* Mistral-Base:
```shell
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file accelerate_configs/deepspeed_zero3.yaml scripts/run_simper.py training_configs/mistral-7b-base-simper.yaml
```
* Mistral-Instruct:
```shell
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file accelerate_configs/deepspeed_zero3.yaml scripts/run_simper.py training_configs/mistral-7b-instruct-simper.yaml
```
* Llama3-Base:
```shell
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file accelerate_configs/deepspeed_zero3.yaml scripts/run_simper.py training_configs/llama-3-8b-base-simper.yaml
```
* Llama3-Instruct:
```shell
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file accelerate_configs/deepspeed_zero3.yaml scripts/run_simper.py training_configs/llama-3-8b-instruct-simper.yaml
```

## Evaluation

We follow the official implementation for evaluation on AlpacaEval 2, MT-Bench and Open LLM Leadboard (v1&v2), as follows:

* AlpacaEval 2: Please refer to the [AlpacaEval repo](https://github.com/tatsu-lab/alpaca_eval) for evaluation.

* MT-Bench: Please refer to the [FastChat repo](https://github.com/lm-sys/FastChat) for evaluation.

* Open LLM Leadboard (v1): Please refer to to the [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/b281b0921b636bc36ad05c0b0b0763bd6dd43463) and [Open LLM Leadboard v1](https://huggingface.co/spaces/open-llm-leaderboard-old/open_llm_leaderboard) for evaluation.

* Open LLM Leadboard (v2): Please refer to to the [Language Model Evaluation Harness](https://github.com/huggingface/lm-evaluation-harness/tree/adding_all_changess) and [Open LLM Leadboard v2](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard)  for evaluation.


## Acknowledgement
We build our project based on
- [SimPO](https://github.com/princeton-nlp/SimPO/tree/main?tab=readme-ov-file)
- [The Alignment Handbook](https://github.com/huggingface/alignment-handbook)
- [TRL-Transformer Reinforcement Learning](https://github.com/huggingface/trl)

##  Reference 

If you find our repo to be useful, please cite our paper:
```bibtex
@inproceedings{SimPER2025,
  title={SimPER: A Minimalist Approach to Preference Alignment without Hyperparameters},
  author={Xiao, Teng; Yuan, Yige; Chen, Zhengyu; Li, Mingxiao; Liang, Shangsong; Ren, Zhaochun; Honavar, Vasant G.},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025}
}

