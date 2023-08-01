# Awesome Llama2 Resources

Llama2 is a part open source commercial model released from Meta, including 7B/13B/70B and chat models with 4096 context window.

## Models
- [Original Model] 202307 Meta Released Llama2
    - [Github](https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md)
    - [Paper: Llama 2: Open Foundation and Fine-Tuned Chat Models](https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/)
    - [Download Applications](https://ai.meta.com/resources/models-and-libraries/llama-downloads/)
- [Togehter AI] 202307 TogetherAI released [Llama2-7B context window with 32k](https://huggingface.co/togethercomputer/LLaMA-2-7B-32K) context window based on Meta's research [Extending Context Window of Large Language Models via Positional Interpolation](https://arxiv.org/abs/2306.15595)
    - [Together.ai's blog about Preparing for the era of 32K context: Early learnings and explorations](https://together.ai/blog/llama-2-7b-32k)
## Demo
- [Llama2 70B Chatbot at HuggingFace](https://huggingface.co/spaces/ysharma/Explore_llamav2_with_TGI)

## Finetune Method/ Scripts
- [Finetune with PEFT](https://huggingface.co/blog/llama2?fbclid=IwAR2G3jtbsUMZCTNsYTuxKDJCC_S6SuyFBk8hs0y23TI2ndPHVZ33ZWNHfSc)
- [Finetune together.ai 32k context window model](https://github.com/togethercomputer/OpenChatKit/tree/main/training): script to finetune on booksum/mqa dataset
- [Finetune with QLora at 13b model](https://colab.research.google.com/drive/16SlGXLuBRB30clB0dCYAh3sqk0edKoFC?usp=drive_link&fbclid=IwAR2mXiV4IK0PnuhbozWtkMQANiU6P2u6h03reYtDLMctK3GeM2xEyYQvjNw): a colab about finetuning llama2
- [HuggingFace SFT training script](https://github.com/lvwerra/trl/blob/main/examples/scripts/sft_trainer.py)
- [Pytorch-lightening's script to finetune Llama2 on custom dataset](https://lightning.ai/blog/how-to-finetune-gpt-like-large-language-models-on-a-custom-dataset/)

## Porting
- [Karpathy's Llama2.c](https://github.com/karpathy/llama2.c): Karpathy's weekend project to build a LLama2 at C 

- [web-llm](https://github.com/mlc-ai/web-llm): Bringing large-language models and chat to web browsers

## Tutorial
- [Llama2.c for dummies](https://github.com/RahulSChand/llama2.c-for-dummies): a description about Karpathy's LLama2 line by line

## Prompt
- [LLama2 prompt template](https://gpus.llm-utils.org/llama-2-prompt-template/)

## For specific usage Model/ Finetuned model
- [Huggingface trend about llama2](https://huggingface.co/models?p=13&sort=trending&search=llama2)
- [Chinese-Llama-2-7b](https://github.com/LinkSoul-AI/Chinese-Llama-2-7b): finetune on a chinese and english instruction dataset with 10 millions size
- [Chinese-LLaMA-Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca)
- [Finetuned on code with qLoRA](https://www.linkedin.com/posts/abhishek-harshvardhan-mishra_tokenbenderllama2-7b-chat-hf-codecherrypop-activity-7088091401397145600-FndC/?utm_source=share&utm_medium=member_ios)

## Multimodal LLM
- [LLaSM: Large Language and Speech Model](https://github.com/LinkSoul-AI/LLaSM): Support chinese/english voice chat model based on whisper features
- [LLaVA ](https://github.com/haotian-liu/LLaVA): Large Language-and-Vision Assistant 
- [Chinese-LLaVA](https://github.com/LinkSoul-AI/Chinese-LLaVA): support vision input and chinese text input/output

## Toolkits
- [TogetherAI] [OpenChatKit](https://github.com/togethercomputer/OpenChatKit): Together.ai's open toolkit for LLM finetune/moderation
- [LLaMA2-Accessory](https://github.com/Alpha-VLLM/LLaMA2-Accessory):An Open-source Toolkit for LLM Development 
    - [LLaMA-Adapter](https://github.com/OpenGVLab/LLaMA-Adapter): Fine-tuning LLaMA to follow Instructions within 1 Hour and 1.2M Parameters 

## Other Resources
- [Llama 2 の情報まとめ](https://note.com/npaka/n/ncc6c32fcfd38)
- [LLaMA2-Every Resource you need](https://www.philschmid.de/llama-2)
