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
- [Codellama](https://github.com/facebookresearch/codellama): Meta finetuned Llama2 for code generation usage. Support  C++/ Java/ PHP/ Type Script/ C#/ Bash/ Python generation. Include models 7B/13B/34B，and 3 kind of variation (Generatl/python/instruction). Extend maximum context window from 4,096 tokens to 100k(like claude2).

## Demo
- [Llama2 70B Chatbot at HuggingFace](https://huggingface.co/spaces/ysharma/Explore_llamav2_with_TGI)
- [A16z's Llama2-chatbot](https://github.com/a16z-infra/llama2-chatbot): provide a streamlit chatbot app for LLaMA2
## Finetune Method/ Scripts
- [Finetune with PEFT](https://huggingface.co/blog/llama2?fbclid=IwAR2G3jtbsUMZCTNsYTuxKDJCC_S6SuyFBk8hs0y23TI2ndPHVZ33ZWNHfSc)
- [Finetune together.ai 32k context window model](https://github.com/togethercomputer/OpenChatKit/tree/main/training): script to finetune on booksum/mqa dataset
    - [Llama-2-7B-32K-Instruct — and fine-tuning for Llama-2 models with Together API](https://together.ai/blog/llama-2-7b-32k-instruct): Together AI show their 32k context instruct 7b model.
- [Finetune with QLora at 13b model](https://colab.research.google.com/drive/16SlGXLuBRB30clB0dCYAh3sqk0edKoFC?usp=drive_link&fbclid=IwAR2mXiV4IK0PnuhbozWtkMQANiU6P2u6h03reYtDLMctK3GeM2xEyYQvjNw): a colab about finetuning llama2
- [HuggingFace SFT training script](https://github.com/lvwerra/trl/blob/main/examples/scripts/sft_trainer.py)
- [Pytorch-lightening's script to finetune Llama2 on custom dataset](https://lightning.ai/blog/how-to-finetune-gpt-like-large-language-models-on-a-custom-dataset/)
- [Instuction-tune Llama2](https://www.philschmid.de/instruction-tune-llama-2): HuggingFace's Tech Lead Philschmid introduced how to instruct finetune Llama2
- [Finetune LLaMA2 7-70B on Amazon SageMaker](https://www.philschmid.de/sagemaker-llama2-qlora): Philschmid introduce preparing datasets/using QLoRA/Deploy model on Amazon SageMaker
- [Finetune LLaMa2 with QLoRA at colab](https://colab.research.google.com/drive/1Zmaceu65d7w4Tcd-cfnZRb6k_Tcv2b8g?usp=sharing#scrollTo=AgKCL7fTyp9u)
- [Fine-tune Llama 2 with DPO by huggingface](https://huggingface.co/blog/dpo-trl)
- [Fine-tune Llama2 on specific usage like SQL Gen/Functional Representation](https://www.anyscale.com/blog/fine-tuning-llama-2-a-comprehensive-case-study-for-tailoring-models-to-unique-applications): Anyscale's member used their lib `ray` to demo finetune Llama2 70B.[Their scripts](https://github.com/ray-project/ray/tree/master/doc/source/templates/04_finetuning_llms_with_deepspeed)
## Porting
- [Karpathy's Llama2.c](https://github.com/karpathy/llama2.c): Karpathy's weekend project to build a LLama2 at C 

- [web-llm](https://github.com/mlc-ai/web-llm): Bringing large-language models and chat to web browsers
- [HuggingFace release Swift Transformers to help run LLM on Apple Device](https://huggingface.co/blog/swift-coreml-llm): Provide Swift based [Swift Transformers Lib](https://github.com/huggingface/swift-transformers), a [swift chat app](https://github.com/huggingface/swift-chat) and a [exporters](https://github.com/huggingface/exporters) for exporting model to coreml.
- [pyllama](https://github.com/juncongmoo/pyllama): LLaMA: Open and Efficient Foundation Language Models

## Tutorial
- [Meta's started guide to use Llama](https://ai.meta.com/llama/get-started/?utm_source=facebook&utm_medium=organic_social&utm_campaign=llama2&utm_content=image&fbclid=IwAR1vbWQ3ns9atHjIqm6X6gL0DtMk-39jXctSWcq4IAeAJluqe7QCi7PJhP0)
- [Llama2.c for dummies](https://github.com/RahulSChand/llama2.c-for-dummies): a description about Karpathy's LLama2 line by line
- [NeurIPS 2023 LLM Efficiency Challenge Quickstart Guide](https://github.com/Lightning-AI/lit-gpt/blob/main/tutorials/neurips_challenge_quickstart.md): A competition focused on training 1 LLM for 24 hours on 1 GPU – the team with the best LLM gets to present their results at NeurIPS 2023.

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
- [Taiwan-LLaMa](https://github.com/MiuLab/Taiwan-LLaMa): NTU's MiuLab finetune 13B Llama2 with 5B traditional chinese tokens and 490k instruction dataset.
- [Finetuning LLaMa + Text-to-SQL](https://github.com/run-llama/modal_finetune_sql): LlamaIndex show how to fine-tune LLaMa 2 7B on a Text-to-SQL dataset
## Multimodal LLM
- [LLaSM: Large Language and Speech Model](https://github.com/LinkSoul-AI/LLaSM): Support chinese/english voice chat model based on whisper features
- [LLaVA ](https://github.com/haotian-liu/LLaVA): Large Language-and-Vision Assistant 
- [Chinese-LLaVA](https://github.com/LinkSoul-AI/Chinese-LLaVA): support vision input and chinese text input/output

## Toolkits
- [TogetherAI] [OpenChatKit](https://github.com/togethercomputer/OpenChatKit): Together.ai's open toolkit for LLM finetune/moderation
- [LLaMA2-Accessory](https://github.com/Alpha-VLLM/LLaMA2-Accessory):An Open-source Toolkit for LLM Development 
    - [LLaMA-Adapter](https://github.com/OpenGVLab/LLaMA-Adapter): Fine-tuning LLaMA to follow Instructions within 1 Hour and 1.2M Parameters 
- [text-generation-webui](https://github.com/oobabooga/text-generation-webui):A gradio web UI for running Large Language Models like LLaMA, llama.cpp, GPT-J, OPT, and GALACTICA.
- [text-generation-inference](https://github.com/huggingface/text-generation-inference): Huggingface's Large Language Model Text Generation Inference.
- [FlexFlow Serve: Low-Latency, High-Performance LLM Serving](https://github.com/flexflow/FlexFlow): An open-source compiler and distributed system for low latency, high performance LLM serving.
- [LLM-As-Chatbot](https://stanford-cs221.github.io/autumn2021/):Use lots of open sourced instruction-following fine-tuned LLM models as a Chatbot service.
## Optimiztion(Latency/Size)
- [Optimizing LLM latency](https://hamel.dev/notes/llm/03_inference.html): A great blog about exploration of inference tools for open source LLMs
- [Series Quantized LLama2 Model from The Bloke with GPTQ/GGML](https://huggingface.co/TheBloke)
    - [TheBloke/llama-2-7B-Guanaco-QLoRA-GPTQ](https://huggingface.co/TheBloke/llama-2-7B-Guanaco-QLoRA-GPTQ)
    - [OpenAssistant-Llama2-13B-Orca-8K-3319-GPTQ](https://huggingface.co/TheBloke/OpenAssistant-Llama2-13B-Orca-8K-3319-GPTQ)
- Quantization
    - [GPTQ: Accurate Post Training Quantization for generative pre-trained transformers](https://arxiv.org/pdf/2210.17323.pdf)
        - [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ): An easy-to-use LLMs quantization package with user-friendly apis, based on GPTQ algorithm.
- [Together AI's Medusa to accelerate decoding](https://together.ai/blog/medusa)
- [NVIDIA TensorRT-LLM Supercharges Large Language Model Inference on NVIDIA H100 GPUs](https://developer.nvidia.com/blog/nvidia-tensorrt-llm-supercharges-large-language-model-inference-on-nvidia-h100-gpus):TensorRT-LLM is an open-source library that accelerates and optimizes inference performance on the latest LLMs on NVIDIA Tensor Core GPUs. 
## Optimization(Reasoning)
- [LLM Reasoners](https://github.com/Ber666/llm-reasoners): LLM Reasoners is a library to enable LLMs to conduct complex reasoning, with advanced reasoning algorithms.
- [Deepminds LLM as Optimizers](https://twitter.com/omarsar0/status/1700249035456598391?s=12&t=Uq8kqKilbZENNaHu98Z26A)

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
- [Llama with FlashAttention2](https://twitter.com/birchlabs/status/1692319402006384938?s=12&t=Uq8kqKilbZENNaHu98Z26A): Reduces VRAM usage, especially during training.Full finetune Llama 2 7b:51.3->40.3GiB

- [Anti-hype LLM reading list](https://gist.github.com/veekaybee/be375ab33085102f9027853128dc5f0e): A reading list about LLM.
## Move on to production
- [Patterns for Building LLM-based Systems & Products](https://eugeneyan.com/writing/llm-patterns/): Amazon's LLM Engineer Eugene Yan wrote a blog about patterns of LLM based system
- [Finetuning an LLM: RLHF and alternatives](https://medium.com/mantisnlp/finetuning-an-llm-rlhf-and-alternatives-part-i-2106b95c8087)
- [Github:A developer’s guide to prompt engineering and LLMs](https://github.blog/2023-07-17-prompt-engineering-guide-generative-ai-llms/): Github engineer shares their experiences to to prompt engineering for their copilot product.
- [The Rise and Potential of Large Language Model Based Agents: A Survey](https://arxiv.org/pdf/2309.07864.pdf): A survey from Fudan NLP Group about LLM based Agents. Their github repo https://github.com/WooooDyy/LLM-Agent-Paper-List

## Evaluation
- [🤗Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard): A huggingface space which  track, rank and evaluate LLMs and chatbots as they are released.

## Calculation
- [How is LLaMa.cpp possible](https://finbarr.ca/how-is-llama-cpp-possible/): The post showed why Llama is limited by memory bound with some calculations of the transformers parameters.
    - [Transformer Math101](https://blog.eleuther.ai/transformer-math/)
    - [Transformer inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/)
    - [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)

## Some theory
- [Why we should train smaller LLMs on more tokens](https://www.harmdevries.com/post/model-size-vs-compute-overhead/)
    - harms law on hugging face for calculating the model size/dataset size's compute overhead
- [LLMSurvey](https://github.com/RUCAIBox/LLMSurvey): A Survey of Large Language Models
- [Open challenges in LLM research](https://huyenchip.com/2023/08/16/llm-research-open-challenges.html): Chip Huyen's post about LLM's challenge
- [Stanford CS324 - Large Language Models](https://stanford-cs324.github.io/winter2022/): The fundamentals about the modeling, theory, ethics, and systems aspects of large language models.
    - [CS221:Artificial Intelligence: Principles and Techniques](https://stanford-cs221.github.io/autumn2021/)

- [Why you(Propbably) Don't Need to Fine-tune an LLM](https://www.tidepool.so/2023/08/17/why-you-probably-dont-need-to-fine-tune-an-llm/): Finetuning maynot reduce hallucinating. You could use few-shot prompting/ Retrieval Augmented Generation(RAG)

- [Challenges and Applications of Large Language Models](https://arxiv.org/pdf/2307.10169.pdf)
## Some basics 
-  [Some Intuition on Attention and the Transformer](https://eugeneyan.com/writing/attention/): A post introduces the big deal about attention/what are query,key and value