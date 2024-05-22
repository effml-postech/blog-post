---
type: docs
bookToc: True
weight: 1
---

# Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention

*Team: Younghyun Cho, Sangjun Lee

# Large Language Models with Inifinite Input Sequence

Currently, **Large Language Models (LLM)** are based on **Transformer** architecture [1], which utilize interactions over the segments of an input sequence. However, this architecture has a limitation, requiring **huge computations** and **memory** in proportion to the **length** of input sequences. Thereby, current LLM are **struggled** to **infinite input sequence tasks** like summarizing books.  To overcome this problem, google researchers suggests combining transformer architecture with a **compressive memory**, which **stores previous informations** in a **constant size**. They dubbed this method as *Infini-attention*.

# Detailed Method

## Background: Scaled Dot-product Attention

The **multi-head scaled dot-product attention** (a.k.a. self-attention or MHA) is the main component in transformer architectures’ block. 

To calculate the attention state {{< katex >}} A_{dot} ∈ \mathbb R^{N×d_{value}}  {{< /katex >}}  of a single head in the MHA module with an input seqeunce  {{< katex >}} $X ∈ \mathbb R^{N×d_{model}}$  {{< /katex >}} , three components, key, query and value are computed as 


<p align="center">
    {{< katex >}}
K = XW_K, V = XW_V \ \text{and} \ Q = XW_Q,
  {{< /katex >}} 
</p>

where $W_K ∈ \mathbb R^{d_{model} ×d_{key }}$, $W_V ∈ \mathbb R^{d_{model} ×d_{value}}$ and $W_Q ∈ \mathbb R^{d_{model} ×d_{key}}$ are trainable projection matrices. Then, we can get the attention state as

$$
A_{dot} = \text{softmax} \Bigl(  \dfrac {QK^\top}  {\sqrt {d_{model}}}  \Bigr) V.
$$

We could calculate the MHA By parallely computing $H$ number of attention states over an input sequence and then concatenating them.

## Infini-attention

<p align="center">
    <img src=./Untitled.png>
</p>
<p align="left" style="color:gray">
Figure 1: Infini-attention has an additional compressive memory with linear attention for processing infinitely long contexts. $\{KV\}_{s−1}$ and $\{KV\}_s$ are attention key and values for current and previous input segments, respectively and $Q_s$ the attention queries. PE denotes position embeddings. 
</p>

As shown Figure 1, Infini-attention computes both **local** and **global context** states and **combine them** for its output. Similar to multi-head attention (MHA), it maintains $H$ number of parallel compressive memory per attention layer ($H$ is the number of attention heads) in addition to the dot-product attention.

## Compressive Memory

The researchers suggest three implementation to properly maintain the compressive memory inspired from previous neural network memory [11, 12, 13].

### Memory Retrieval

To **fetch** information **from the memory**, **Infini-attetnion** simply **reuse the query** **($Q$)** **in current state** and **combine with the memory**. Specifically, the attention state from the memory $M_{s−1} ∈ \mathbb R^{d_{key} ×d_{value }}$, $A_{mem} ∈ \mathbb R^{N×d_{value}}$, is computed as follows with query $Q ∈ \mathbb R^{N×d_{key}}$:

$$
A_{mem} = \dfrac {σ(Q)M_{s−1}}{σ(Q)z_{s−1}}.
$$

$\sigma$ is a nonlinear activation function, and $z_{s−1} ∈ \mathbb R^{d_{key}}$ is a normalization term. The researchers use element-wise ELU+1 and sum over all keys for each described before.

### Memory Update

After the retrieval of the memory, we should **update the memory** and **normalization part** with **the current key** and **value** as follows: 

$$
M_s ← M_{s−1} + σ(K)^\top V \ \text{and} \ z_s ← z_{s−1}  + \sum^N_{t=1}σ(K_t). 
$$

After the update, the **next input segment $S+1$** uses the **updated memory** $M_s$ and **normalization term** $z_s$ recursively. Also, $σ(K)^\top V$ is refered to associative binding operator [3].

Also, the authors combines the delta rule [2] into Infini-attention. The delta rule takes the difference between the value of new segment ($V$) and the stored value in memory as the associative binding terms instead of simply using $V$ (which is similar as the *advantage function* in reinforcement learning).

$$
M_s ← M_{s−1} + σ(K)^\top (V − \dfrac {σ(K)M_{s−1}} {σ(K)z_{s−1}}). 
$$

The authors call this method as $Linear+Delta$ and the former method as $Linear$.

### Long-term Context Injection

It is important to have **a balance** in **the local attention $A_{dot}$** and **the global context** $A_{mem}$. The researchers add a scalar $\beta$ which is the gating component of the weighted sum over the above attention states:

$$
A = sigmoid(β) ⊙ A_{mem} + (1 − sigmoid(β)) ⊙ A_{dot}. 
$$

Finally, to get the MHA output of an attention layer $O ∈ \mathbb R^{N×d_{model }}$, we concatenate the $H$ parallel attention state and then project them to the output dimension:

$$
O = [A^1; . . . A^H ]W_O
$$

where $W_O ∈ \mathbb R^{H×d_{value} ×d_{model}}$ is the projection weights.

## Comparsion with Other Transformers with Context Memory

<p align="center">
    <img src=./Untitled%201.png>
</p>

<p align="left" style="color:gray">
Table 1: Transformer models with segment-level memory are compared. For each model, the memory size and effective context length are defined in terms of their model parameters ($N$: input segment length, $S$: the number of segments, $l$: the number of layers, $H$: the number of attention heads, $c$: Compressive Transformer memory size, $r$: compression ratio, $p$: the number of soft-prompt summary vectors and $m$: summary vector accumulation steps).
</p>

Table 1 shows the analysis of transformer models combining with segment-level memory. 

- **Transformer-XL** [4] **uses KV components from the privious segment** with **current components** over each layer. Thus the context window of Transformer-XL is enlarged from $N$ to $N \times L$, and it requires  $(d_{key} + d_{value}) × H × N × l$ memory foot prints.
<p align="center">
<img src=./Untitled%202.png>
</p>
    
<p align="center" style="color:gray">
Figure from Transformer-XL [4]. Illustration of the vanilla model with a segment length 4.
</p>

- **Compressive Transformer** [5] append **additional cache** to Transformer-XL that **saves the past activations**. It broaden the Transformer-XL’s context window by $c × r × l$. It keeps a fine-grained memory of past activations, which are then compressed into coarser compressed memories. The below model has three layers, a  sequence length $n_s = 3$, memory size $n_m = 6$, compressed memory size $n_{cm} = 6$. The highlighted memories are compacted, with a compression function $f_c$ per layer, to a single compressed memory — instead of being discarded at the next sequence. In this example, the rate of compression $c = 3$.
<p align="center">
<img src=./Untitled%203.png>
</p>   

<p align="center" style="color:gray">
Figure from Compressive Transformer [5]. 
</p>

    
- **Memorizing Transformers** [6] trys to **gather the every KV components** as the global context for the input segment. To reduce the overhead of storing every KV compoents, Memorizing Transformers adapts the context-weaving only on the last layer. The context window could explore entire input sequence $N \times S$ using KNN retriever.
<p align="center">
<img src=./Untitled%204.png>
</p>
    
<p align="left" style="color:gray">
Figure from Memorizing Transformers [6]. Memorizing Transformers extend Transformers with access to (key, value) pairs of previously seen subsequences.
</p>
    
- **RMT** [7] and **AutoCompressors** [8, 9] utilized **extra vectors** that **interact with current segment** and then **is delivered to next token** recursively (which is similar in hidden vector in Recurrent Neural Networks (RNN)). However, the google researchers argue that the size of the additional memory vectors is the main factor of the efficiency of the method, which means that the performance and the memory footprint is aligned each other.
    
<p align="center">
<img src=./Untitled%205.png>
</p>
    
<p align="left" style="color:gray">
Figure from Recurrent Memory Transformer [7]. Memory is added as tokens to the input sequence and memory output is passed to the next segment. During training gradients flow from the current segment through memory to the previous segment.
</p>

<p align="center">
<img src=./Untitled%206.png>
</p>

<p align="left" style="color:gray">
Figure from AutoCompressors [8]. AutoCompressors process long documents by recursively generating summary vectors which are passed as soft prompts to all subsequent segments.
</p>
    
<p align="center">
    <img src=./Untitled%207.png>
</p>

<p align="left" style="color:gray">
Figure 2.  Infini-Transformer (top) has an entire context history whereas Transformer-XL (bottom) discards old contexts since it caches the KV states for the last segment only.
</p>

Compare to the above context-based transformer models, **Infini-Transformer** could **catch the entire context** $N\times S$ with **the fixed memory size** $d_{key} × d_{value} + d_{key}$ that only stores $M_s$ and $z_s$ over the every attention heads and layers.

# Experiments

Infini attention was tested on three main benchmarks such as **long-context language modeling**, **passkey retrieval** and **book summarization**. 

### Long-context Language Modeling

<p align="center">
    <img src=./Untitled%208.png>
</p>

<p align="center" style="color:gray">
Table 2: Long-context language modeling results are compared in terms of average token-level perplexity. Comp. dentoes compression ratio. Infini-Transformer outperforms memorizing transformers with memory length of 65K and achieves 114x compression ratio.
</p>

The authors trained and evaluated small Infini-Transformer models on **PG19** [5] and
**Arxiv-math** [6] **benchmarks**. They noted that the model with Infini Attention outperformed the baseline model. Additionally, extending the training sequence length further improved the perplexity score, a metric indicating language model performance, where lower scores signify better performance.

<p align="center">
    <img src=./Untitled%209.png>
</p>

<p align="center" style="color:gray">
Figure 3. Gating score visualization.
</p>

Figure 3 illlustrates the gating value ($sigmoid(\beta)$) of each heads and layers of the pretrained Infini-Transformer. The speciallized head means that the gating scores are close to 0 or 1 which only pass the local attention outputs or context attention output from the memory. The mixer head, of which the gating scores is near 0.5, combines the both information.

### 1M passkey retrieval benchmark

<p align="center">
    <img src=./Untitled%2010.png>
</p>

<p align="left" style="color:gray">
Table 3: Infini-Transformers solved the passkey task with up to 1M context length when fine-tuned on 5K legnth inputs. We report token-level retrieval accuracy for passkeys hidden in a different pat (start/middle/end) of long inputs with lengths 32K to 1M
</p>

The **pass-key task** is a task that **hides a random number** in a long context and **asks it back** in the model output. Below is the input format of the passkey task.

*There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there. The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. (repeat x times) The pass key is **9054**. Remember it. **9054** is the pass key. The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. (repeat y times) What is the pass key? The pass key is*

While previous work [14] showed that the 8B LLaMA model can solve tasks up to 32K in length **when fine-tuned** with the same **32K-long inputs using Position Interpolation**, Infini-Transformer takes this problem further, fine-tuning with only 5K-long inputs and testing it on a 1M-long region. They reported both zero-shot accuracy and finetuning accuracy. Table 3 shows that Inifni-Transformer **solved the passkey test perfectly from 32K to 1M after FT.** 

### 500K length book summarization (BookSum)

<p align=center>
    <img src=./Untitled%2011.png>
</p>

<p align="center" style="color:gray">
Table 4: 500K length book summarization (BookSum) results. The BART, PRIMERA and Unlimiformer results are from Bertsch et al. (2024).
</p>

<p align=center>
    <img src="./Untitled%2012.png">
</p>

<p align="center" style="color:gray">
Figure 4: Infini-Transformers obtain better Rouge overall scores with more book text probived as input.
</p>

The researchers scaled up their approach by continuously pre-training an 8B LLM model with an 8K input length over 30K steps. They then **fine-tuned** this model **for the book summarization task**, setting the input length to 32K for fine-tuning and 500K for evaluation. They used a generation temperature of 0.5, a top-p of 0.95, and a decoding step of 1024 to generate book summaries. Their model **outperforms previous best results** and **sets a new state-of-the-art on BookSum** by processing the full text of books. Figure 4 shows the overall Rouge score for the validation split of the BookSum data, indicating a clear trend: **the more text provided from the book, the better the summary performance for Infini-Transformers.**

# Conclusion

This work presents a novel attention, Infini-Attention, which is a close **integration of compressive memory module into the vanilla dot-product attention layer.** It builds both masked local attention and long-term linear attention into a single transformer block. It helps **handle infinitely long processes** with **limited memory and computation resources.** As long-context LLMs are increasingly important today, having such an effective memory system shows the potential for powerful reasoning, planning, continuous adaptation and capabilities not previously seen in LLMs.

# Discussion

- Since Infini-Attention compresses and stores information, **it is questionable** whether it can **produce inconsistent** or **confusing output** if it conflicts with the knowledge of the base model.
- It use the name Infini-Attention due to its incremental updates, but the authors only test it on 1M tokens. As mentioned earlier, **it is doubtful** that it can perform on **truly infinite data** with minimal information loss.
- We can use memory-based not only for language tasks but for the other tasks. For example, In **transformer models** for **videos** [15], they compute over spatio-temporaly combined 3D input (multiple frames) with transformer model, but this requires huge computation overhead. Instead, we could only use a transformer models with **2D input** that only takes **one frame** with  **the compressive memory** that stores **global context** extracted from the previous frames.


# Reference

[1] “Attention Is All You Need.**”,** Vaswani et al.

[2] “Metalearned neural memory.”, Munkhdalai et al.

[3] “Tensor product variable binding and the representation of symbolic structures in connectionist systems.”, Smolensky.

[4] “Transformer-xl: Attentive language models beyond a fixed-length context.”, Dai et al.

[5] “Compressive transformers for long-range sequence modelling.”, Rae et al.

[6] “Memorizing transformers.”, Wu et al.

[7] “Recurrent Memory Transformer.” Bulatov et al.

[8] “Adapting Language Models to Compress Contexts.”, Chvalier et al.

[9] “In-context Autoencoder for Context Compression in a Large Language Model.”, Ge et al.

[10] “Leave No Context Behind: Efficient Infinite Context Transformer with Infini-attention.”, Munkhdalai et al.

[11] “Metalearned neural memory.”, Munkhdalai et al.

[12] “Learning associative inference using fast weight memory.”, Schlag.

[13] “Transformers are rnns: Fast autoregressive transformers with linear attention.”, Katharopoulos et al.

[14] “Extending context window of large language models via positional interpolation.” Chen et al.

[15] “ViViT: A Video Vision Transformer.”, Arnab et al.
