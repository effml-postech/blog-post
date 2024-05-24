---
type: docs
bookToc: True
weight: 1
---

# XC-CACHE: Cross-Attending to Cached Context for Efficient LLM Inference

*Submitted on 23 Apr 2024 by* João Monteiro1, Étienne Marcotte1,*, Pierre-André Noël1,*, Valentina Zantedeschi1,*, David Vázquez1, Nicolas Chapados1, 2, Christopher Pal1, 2, Perouz Taslakian11, ServiceNow Research.

 [In-context learning, ICL](https://www.notion.so/In-context-learning-ICL-817a5a05df7945149e6efbf8972f645c?pvs=21)  typically uses prompting to condition the generation of decoder-only language models based on reference information. However, just-in-time processing of context is inefficient due to the quadratic cost of self-attention operations, making **[caching](https://www.notion.so/7708a6d8eeaf4a7082e9da6b4b90950d?pvs=21)** desirable. Yet, caching [transformer](https://www.notion.so/e10d87b5902d4dfb94edbbecc2582efe?pvs=21) states can demand nearly as much space as the model parameters themselves, posing a challenge when the appropriate context is not known in advance.

 This paper addresses these challenges by introducing models that use **cross-attention**, inspired by the encoder-decoder architecture, to condition generation on reference text without a prompt. The approach leverages pre-trained decoder-only models and trains only a small number of added layers. The authors use Question-Answering (QA) as a testbed to evaluate these models' ability to perform conditional generation. 

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/ecaf2b8d-d654-4cb1-bdb0-0aa6aa6ea46d/aee2a0ac-0dff-4ec3-a990-297af09450ac/Untitled.png)

 These four approaches highlight various strategies for efficient context processing in large language models. 

(a) depicts a scenario where a user’s query must be interpreted within a given context to generate an answer. In this case, the query and answer are small (light), but the context is large (heavy). This results in a time complexity of O(|context|²) for the [LLM](https://www.notion.so/LLM-e5c86372d9404c97ba24cb1491243a84?pvs=21).

(b) explains [***In-Context Learning (ICL)***](https://arxiv.org/abs/2005.14165) and [***Retrieval-Augmented Generation (RAG)](https://arxiv.org/abs/2005.11401)***  which use the query to look up the context from a finite corpus, but still remain inefficient with large contexts.

(c) can be preprocessed into a cache, enabling fast inference on a given query. This [***approach***](https://arxiv.org/abs/1706.03762) has a time complexity of O(|context|(|query| + |answer|)).

(d) is the method that the author proposed named **XC-CACHE**. It is implemented in two ways that leverage pre-trained decoder-only models and add a separate encoder to process the context: one approach uses the frozen decoder as an encoder (called XC-LLAMA), and the other uses a small bidirectional encoder (called XC-LLAMAENC).

## **Caching Representations and XC-CACHING**

[**Cache**](https://en.wikipedia.org/wiki/Cache_(computing)) is typically a form of memory that allows for fast access, storing data that is frequently used or needed repeatedly. [***The main purpose*](https://www.notion.so/7708a6d8eeaf4a7082e9da6b4b90950d?pvs=21)** of a cache is to improve processing speed and enhance the overall efficiency of the system. There are [***three types of elements***](https://www.notion.so/e10d87b5902d4dfb94edbbecc2582efe?pvs=21) called ‘key’, ‘value’, ‘query’ which can approaches to Caching. 

 KV(Key-Value) Caching ******is to store the (past) key and value states generated while processing context.  As an example, for [***LLAMA 2-7B***](https://arxiv.org/abs/2307.09288) using 16 bits precision shows that the smaller per-token cache teh sizes are more desirable. JIT(Just-In-Time Key-Value Caching)-KV Caching is an alternative approach involves storing the (past) hidden states of the model in the cache. At inference time, once these hidden states are loaded on GPU, we can recover the full keys and values in O(|context|). These two KV and JIT-KV Caching model both entail two types of costs while yielding identical results: the size of the cache and the operations required during inference. So **XC-Caching** is presented as an effective way to improve inference speed while significantly reducing memory usage.

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/ecaf2b8d-d654-4cb1-bdb0-0aa6aa6ea46d/4e810569-e761-4ed0-a754-ee658a9fb77e/Untitled.png)

(a) The architecture uses a small bidirectional encoder and multiple self-attention and cross-attention layers to process the context and prompt.
(b) The architecture uses only a decoder, mainly training the cross-attention layers to process the context and prompt.

## Low memory usage, High precision XC-CACHING

 QA is ideal for testing the methods as it requires efficient external information retrieval and incorporation during generation. They focus on training for question-answering using datasets with context, query, and answer triplets. They build a training dataset by standardizing and combining the training partitions of five publicly available and diverse datasets: [***NATURAL QUESTIONS](https://aclanthology.org/Q19-1026/)*** (NQ), [***HOTPOTQA***](https://arxiv.org/abs/1809.09600), [***TOPIOCQA***](https://aclanthology.org/2022.tacl-1.27/), [***MS MARCO***](https://arxiv.org/abs/1611.09268), and [***SQUAD-V2***](https://aclanthology.org/P18-2124/). Each example in the resulting dataset contains a query (natural-language question), an answer (expected output), and one or more contexts (e.g., knowledge base articles), with at least one context containing the answer, referred to as the reference context. 

 In addition to training on the primary QA tasks, they optimize their models on context repetition tasks, named [multitask](https://www.v7labs.com/blog/multi-task-learning-guide) training strategy.
