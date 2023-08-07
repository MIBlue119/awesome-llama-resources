# Awesome Llama2 Resources

Llama2 is a part open source commercial model released from Meta, including 7B/13B/70B and chat models with 4096 context window.

## Models
- [Original Model] 202307 Meta Released Llama2
    - [Github](https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md)
    - [Meta's llama-recipes](https://github.com/facebookresearch/llama-recipes): provide examples for finetuning at SingleGPU/Multiple GPU and the recipe to convert model to HuggingFace transformers's LLama2 model definition
    - [Paper: Llama 2: Open Foundation and Fine-Tuned Chat Models](https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/)
    - [Download Applications](https://ai.meta.com/resources/models-and-libraries/llama-downloads/)
- [Togehter AI] 202307 TogetherAI released [Llama2-7B context window with 32k](https://huggingface.co/togethercomputer/LLaMA-2-7B-32K) context window based on Meta's research [Extending Context Window of Large Language Models via Positional Interpolation](https://arxiv.org/abs/2306.15595)
    - [Together.ai's blog about Preparing for the era of 32K context: Early learnings and explorations](https://together.ai/blog/llama-2-7b-32k)
## Demo
- [Llama2 70B Chatbot at HuggingFace](https://huggingface.co/spaces/ysharma/Explore_llamav2_with_TGI)
- [A16z's Llama2-chatbot](https://github.com/a16z-infra/llama2-chatbot): provide a streamlit chatbot app for LLaMA2
## Finetune Method/ Scripts
- [Finetune with PEFT](https://huggingface.co/blog/llama2?fbclid=IwAR2G3jtbsUMZCTNsYTuxKDJCC_S6SuyFBk8hs0y23TI2ndPHVZ33ZWNHfSc)
- [Finetune together.ai 32k context window model](https://github.com/togethercomputer/OpenChatKit/tree/main/training): script to finetune on booksum/mqa dataset
- [Finetune with QLora at 13b model](https://colab.research.google.com/drive/16SlGXLuBRB30clB0dCYAh3sqk0edKoFC?usp=drive_link&fbclid=IwAR2mXiV4IK0PnuhbozWtkMQANiU6P2u6h03reYtDLMctK3GeM2xEyYQvjNw): a colab about finetuning llama2
- [HuggingFace SFT training script](https://github.com/lvwerra/trl/blob/main/examples/scripts/sft_trainer.py)
- [Pytorch-lightening's script to finetune Llama2 on custom dataset](https://lightning.ai/blog/how-to-finetune-gpt-like-large-language-models-on-a-custom-dataset/)
- [Instuction-tune Llama2](https://www.philschmid.de/instruction-tune-llama-2): HuggingFace's Tech Lead Philschmid introduced how to instruct finetune Llama2
- [Finetune LLaMA2 7-70B on Amazon SageMaker](https://www.philschmid.de/sagemaker-llama2-qlora): Philschmid introduce preparing datasets/using QLoRA/Deploy model on Amazon SageMaker
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
- [ToolLLaMA](https://github.com/OpenBMB/ToolBench): An open source project to train LLaMa on ToolBench, to make LLaMa support function call
- [Llama2-Code-Interpreter](https://github.com/SeungyounShin/Llama2-Code-Interpreter): make Llama2 use Code Execution, Debug, Save Code, Reuse it, Access to Internet 
- [Llama2-Medical-Chatbot](https://github.com/AIAnytime/Llama2-Medical-Chatbot): A medical bot built using Llama2 and Sentence Transformers
- [Finetune LLaMA 7B with Traditional Chinese instruction datasets](https://github.com/A-baoYang/alpaca-7b-chinese)

## Multimodal LLM
- [LLaSM: Large Language and Speech Model](https://github.com/LinkSoul-AI/LLaSM): Support chinese/english voice chat model based on whisper features
- [LLaVA ](https://github.com/haotian-liu/LLaVA): Large Language-and-Vision Assistant 
- [Chinese-LLaVA](https://github.com/LinkSoul-AI/Chinese-LLaVA): support vision input and chinese text input/output

## Toolkits
- [TogetherAI] [OpenChatKit](https://github.com/togethercomputer/OpenChatKit): Together.ai's open toolkit for LLM finetune/moderation
- [LLaMA2-Accessory](https://github.com/Alpha-VLLM/LLaMA2-Accessory):An Open-source Toolkit for LLM Development 
    - [LLaMA-Adapter](https://github.com/OpenGVLab/LLaMA-Adapter): Fine-tuning LLaMA to follow Instructions within 1 Hour and 1.2M Parameters 
## Optimiztion(Latency/Size)
- [Optimizing LLM latency](https://hamel.dev/notes/llm/03_inference.html): A great blog about exploration of inference tools for open source LLMs
- [Series Quantized LLama2 Model from The Bloke with GPTQ/GGML](https://huggingface.co/TheBloke)
    - [TheBloke/llama-2-7B-Guanaco-QLoRA-GPTQ](https://huggingface.co/TheBloke/llama-2-7B-Guanaco-QLoRA-GPTQ)
    - [OpenAssistant-Llama2-13B-Orca-8K-3319-GPTQ](https://huggingface.co/TheBloke/OpenAssistant-Llama2-13B-Orca-8K-3319-GPTQ)
- Quantization
    - [GPTQ: Accurate Post Training Quantization for generative pre-trained transformers](https://arxiv.org/pdf/2210.17323.pdf)
        - [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ): An easy-to-use LLMs quantization package with user-friendly apis, based on GPTQ algorithm.

## Optimization(Reasoning)
- [LLM Reasoners](https://github.com/Ber666/llm-reasoners): LLM Reasoners is a library to enable LLMs to conduct complex reasoning, with advanced reasoning algorithms.

## Use 
- [Run Llama 2 on your own Mac using LLM and Homebrew](https://simonwillison.net/2023/Aug/1/llama-2-mac/)
- [Deploy Llama2 7B/13B/70B model on AWS SageMaker](https://www.philschmid.de/sagemaker-llama-llm): Based on Hugging Face LLM DLC(Deep Learning Container) which is powered by [huggingface's text generation inference](https://github.com/huggingface/text-generation-inference). HuggingFace's text generation inference is a Rust, Python and gRPC server for text generation inference. Used in production at HuggingFace to power Hugging Chat, the Inference API and Inference Endpoint.
## Other Resources
- [Llama 2 の情報まとめ](https://note.com/npaka/n/ncc6c32fcfd38)
- [LLaMA2-Every Resource you need](https://www.philschmid.de/llama-2)
- [LLaMA-efficient-tuning](https://github.com/hiyouga/LLaMA-Efficient-Tuning): Easy-to-use fine-tuning framework using PEFT (PT+SFT+RLHF with QLoRA) (LLaMA-2, BLOOM, Falcon, Baichuan) 
- [awesome-llm and aigc](https://github.com/sjinzh/awesome-llm-and-aigc)
- [Finetune Falcon-7B on Your GPU with TRL and QLoRA](https://medium.com/@bnjmn_marie/fine-tune-falcon-7b-on-your-gpu-with-trl-and-qlora-4490fadc3fbb): A blog about tuning falcon-7b on your consumer GPU
- [A Definitive Guide to QLoRA: Fine-tuning Falcon-7b with PEFT](https://medium.com/@amodwrites/a-definitive-guide-to-qlora-fine-tuning-falcon-7b-with-peft-78f500a1f337)
- [Amazon sagemaker generativeai: Fine-tune Falcon-40B with QLoRA](https://github.com/aws-samples/amazon-sagemaker-generativeai/blob/main/studio-notebook-fine-tuning/falcon-40b-qlora-finetune-summarize.ipynb)
## Some theory
- [Why we should train smaller LLMs on more tokens](https://www.harmdevries.com/post/model-size-vs-compute-overhead/)
    - harms law on hugging face for calculating the model size/dataset size's compute overhead

## Move on to production
- [Patterns for Building LLM-based Systems & Products](https://eugeneyan.com/writing/llm-patterns/): Amazon's LLM Engineer Eugene Yan wrote a blog about patterns of LLM based system