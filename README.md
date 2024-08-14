# llm.pth
Implementation of various Autoregressive models, Research papers and techniques. Main aim is to write clean, modular, wrapper-free implementations.

```bash

## Research log

2024-08-08
----------
formed a new architecture of LLMs which consists of 16k peer experts with 1024 shared 
experts. Kendrick consists of MLHA for efficient KV cache and layer sharing concept. 
Problem with Kendrick is that it doesnt scale above 240M on A40 gpu. Have to solve that!

2024-07-26
----------
implemented Mixture of Million experts paper from Google Deepmind (https://web3.arxiv.org/abs/2407.04153). 
Peer arch is not that much efficient. Maybe a custom cuda kernel for it would be better.

2024-07-15
----------
tried out Dynamic token pruning for llm inference from 
Lazyllm paper (https://arxiv.org/abs/2407.14057)


2024-07-02
----------
implemented Multi-Latent Head Attention (MLA) architecture (https://arxiv.org/abs/2405.04434). 
Its KV cache is equal to GQA with only 2.25 groups, but its performance is stronger than MHA.

2024-06-25
----------
support for new Metas paper regarding Contextual position embedding added. 
(https://arxiv.org/pdf/2405.18719)


2024-06-17
----------
yaRN position embedding (https://arxiv.org/abs/2309.00071) added. Ramp function, attention scaling, and interpolation of specific frequenceis aded. 
Medium blog also added regarding NTK, Linear RoPE, YaRN (https://medium.com/@zaiinn440/linear-rope-vs-ntk-vs-yarn-vs-cope-d33587ddfd35).

2024-06-11
----------
neural Tangent Kernel (NTK) and Linear RoPE support added. Only difference is of change of theta.                             
                            
2024-06-04
----------
mobileLLM architecture implemented. (https://arxiv.org/abs/2402.14905). Swiglu, layer sharing, embedding sharing, GQA used.

2024-05-27
----------
gemma2, llama3 and qwen2 architecture implemented. Qwen chunkllama for context length added.


2024-05-18
----------
mixture of Depths paper implemented.


2024-05-13
----------
arctic architecture of Snowflake implemented and Wandb blog posted.

2024-05-08
----------
dPO/kTO/iPO script added. Data prep for argilla/dpo-mix-7 added. 
Currently single gpu, and no quantization supported.

2024-05-05
----------
sft data preprocessing script added. Took alot of time to get the data in right format 
and then into the cycle iterator for training. SFT script is working. 
Lora/Dora/Qlora not integrated with training script yet.

2024-05-04
----------
tried replacing Linear layers in Llama model with Kolmogorov-Arnold Network layers. 
Total model param of Llama model with linear layers shrank 
from 75M to 47.5M by using KAN. Training script yet to 
be updated to add support kanLlama training.

2024-05-02
----------
explored OLMo, litgpt sft pipelines. Found out the loss function is same, just a little difference in the inputs to CSE. Exploring the data prep for sft different formats as well.

2024-05-01
----------
added LoRA and Dora. Looking at Olmo sft trainer implementation now.

2024-04-26
----------
want to add SFT. Trying to find out its from scratch implementation. Nothing found except 
transformers trainer. TRL only extends this and calls .train for SFT. 
Is there a different loss function for SFT or same as pre-training? IDK

2024-04-27
----------
snowflakes new Hybrid dense-moe architecture added.

2024-04-26
----------
pre-training script added. Training with Fabric is running fine.

2024-04-25
----------
added LLM architectures i.e. Llama, phi, mixtral.

```





## Setup


Let's get this thing running! Follow the next steps:

1. `git clone https://github.com/abideenml/llm.pth.git`
2. Navigate into project directory `cd llm.pth`
3. Create a new venv environment and run `pip install -e .`
4. Run the `llm/utils/prepare-dataset.py` file for data downloading and tokenization.
5. For pre-training, run `python llm/train/pretrain.py`.

That's it!<br/>

## Features

This repo supports various LLM architectures, pretraining, and fine-tuning techniques 

#### ✅ Supported Architectures
* Llama
* Mixtral
* Phi3
* Qwen2
* Deepseekv2
* Gemma2
* Arctic
* MobileLM

#### ✅ Training Techniques
* Pretraining
* LoRA
* QLoRA
* DoRA
* DPO
* KTO
* IPO

#### ✅ Experiments
* Contextual Position Embedding
* LazyLLM
* Multi-Latent Head Attention
* YaRN, NTK
* Mixture of Depths
* Mixture of Million Experts
* Combined architecture of Mome, Mobilelm, deepseekmoe

## 🤞 Todos

Finally there are a couple more todos which I'll hopefully add really soon:
* ORPO
* PPO
* Rejection sampling
* Add yaml config type training like Axolotl, so I don't have to rawdog sft and other techniques
* Evals like ARC, sciq, and more


## 🦋 Citation

If you find this code useful, please cite the following:

```
@misc{Zain2024llm.pth,
  author = {Zain, Abideen},
  title = {llm.pth},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/abideenml/llm.pth}},
}
```

## Connect with me

If you'd love to have some more AI-related content in your life :nerd_face:, consider:

* Connect and reach me on [LinkedIn](https://www.linkedin.com/in/zaiinulabideen/) and [Twitter](https://twitter.com/zaynismm)
* Follow me on 📚 [Medium](https://medium.com/@zaiinn440)
* Check out my 🤗 [HuggingFace](https://huggingface.co/abideen)