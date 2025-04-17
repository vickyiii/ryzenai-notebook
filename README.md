# ryzenai-notebook

Welcome to the Gen AI Playground, a collection of notebooks designed to showcase generative AI workloads on AMD Ryzen AI PCs. These notebooks are designed to be accessible to a broad audience, including AI creators, artists, engineers, and those who are just curious about generative AI.

## Table of Notebook Contents

### LLM

* RAG

  - [Implementing RAG using Lanchain on AI PC](https://github.com/vickyiii/ryzenai-notebook/blob/main/llm/rag/st_rag_lemonade.py)
* Chat

  - [CLI Chatbox Using Lemonade API on AI PC](https://github.com/vickyiii/ryzenai-notebook/blob/main/llm/chat/chat_hybrid.py)
* Search

  - [Search App Using Lemonade API on AI PC](https://github.com/vickyiii/ryzenai-notebook/blob/main/llm/search/search_hybrid.py)

### Hardware

[AMD® Ryzen AI 300 or Ryzen AI Max](https://www.amd.com/zh-cn/products/processors/consumer/ryzen-ai.html#tabs-9f9c97e306-item-6e04e82b39-tab)

## Requirements

The Ryzen AI Software supports AMD processors with a Neural Processing Unit (NPU). Consult the release notes for the full list of supported configurations.

The following dependencies must be present on the system before installing the Ryzen AI Software:

| Dependencies          | Version Requirement |
| --------------------- | ------------------- |
| Windows 11            | build >= 22621.3527 |
| Visual Studio         | 2022                |
| cmake                 | version >= 3.26     |
| Anaconda or Miniconda | Latest version      |

## Installation Steps

1. [Install NPU Drivers](https://ryzenai.docs.amd.com/en/latest/inst.html#install-npu-drivers)
2. [Install Ryzen AI Software 1.4.0](https://ryzenai.docs.amd.com/en/latest/inst.html#install-ryzen-ai-software)
3. [Install TurnkeyML and Lemonade SDK &gt;= 6.1.4](https://github.com/onnx/turnkeyml/blob/main/docs/lemonade/README.md)

## Featured LLMs

The following tables contain a curated list of LLMs that have been validated on Ryzen AI hybrid execution mode. The hybrid examples are built on top of OnnxRuntime GenAI (OGA).

The comprehensive set of pre-optimized models for hybrid execution used in these examples are available in the [AMD hybrid collection on Hugging Face](https://huggingface.co/collections/amd/ryzenai-14-llm-hybrid-models-67da31231bba0f733750a99c). It is also possible to run fine-tuned versions of the models listed (for example, fine-tuned versions of Llama2 or Llama3). For instructions on how to prepare a fine-tuned OGA model for hybrid execution, refer to [Preparing OGA Models](https://ryzenai.docs.amd.com/en/latest/oga_model_prepare.html).

## Notebook Contribution Template

1. Required pip packages
2. Main Application Logic
3. Deployment details with Local LLM
4. Performance Analysis

