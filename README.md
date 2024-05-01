# llm.pth
Implementation of various Autoregressive models, Research papers and techniques. Main aim is to write clean, modular, wrapper-free implementations.

```bash

## Research log

2024-05-01
----------
Added LoRA and Dora. Looking at Olmo sft trainer implementation now.

2024-04-26
----------
Want to add SFT. Trying to find out its from scratch implementation. Nothing found except transformers trainer. TRL only extends this and calls .train for SFT. Is there a different loss function for SFT or same as pre-training? IDK

2024-04-27
----------
Snowflakes new Hybrid dense-moe architecture added.

2024-04-26
----------
Pre-training script added. Training with Fabric is running fine.

2024-04-25
----------
Added LLM architectures i.e. Llama, phi, mixtral

```





## Setup


Let's get this thing running! Follow the next steps:

1. `git clone https://github.com/abideenml/llm.pth.git`
2. Navigate into project directory `cd path_to_repo`
3. Create a new venv environment and run `pip install -e .`
4. Run the `llm/utils/prepare-dataset.py` file for data downloading and tokenization.
5. For pre-training, run `llm/train/pretrain.py`.

That's it!<br/>

## Todos:

Finally there are a couple more todos which I'll hopefully add really soon:
* LoRA and Dora
* DPO/kto/ipo
* Orpo
* mod


## Citation

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