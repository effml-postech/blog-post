---
type: docs
bookToc: True
weight: 1
---

# MIXTURE OF LORA EXPERTS

LoRA is a methodology for effective fine-tuning large-scale pretrained models. LoRA is characterized by its ease of applying tuned results to existing models. This property encouragees research into synthesizing multiple trained LoRAs to achieve enhanced performance across various tasks. Mixture of LoRA Experts (MoLE) presents a new method for achieving the optimal combination of LoRAs for specific tasks. MoLE considers each LoRA as an expert and determines the weights applied to each LoRA at each layer through a gate function.


<p align="center">
    <img src="./mole.png">
    <br>
    <em>Workflow of MoLE</em>
</p>

<!--
제목으로 바꾸기
처음에 어그로 끌 수 있는 내용 먼저
-->
## Background

### What is LoRA?
_Low-Rank Adaptation (LoRA) is a parameter-efficient and effective approach for fine-tuning large-scale pretrained models._

Models such as OPT, LLaMA, and CLIP demonstrate remarkable performance when fine-tuned for various downstream tasks. However, full fine-tuning of these massive models requires substantial computational resources. LoRA enables parameter-efficient fine-tuning by keeping the pretrained model's weights frozen and adding trainable low-rank decomposition matrices.

<p align="center">
    <img src="./LoRA2.png" width="40%">
    <br>
    <em>LoRA Methodology</em>
</p>


In the above figure, only the matrices A and B are trained, with dimensions (d x r) and (r x d) respectively. By setting r << d, the number of parameters to be trained can be reduced. These trained matrices are then simply added to the existing pretrained weights, allowing tuning without affecting the inference speed of the original model.

### LoRAs Composistion

The common solution to further improve the performance of LoRA across various task is to compose multiple trained LoRAs. Research on LoRA composition can be broadly categorized into the following two methodologies.

**Linear arithmetic composition.** It is a method of directly adding multiple LoRAs. This approach is simple and has been effective in the NLP and Vision-Language domain, but it can result in the loss of pre-trained model's generative capabilities or the individual characteristics of each LoRA.

{{< katex display=true >}}
$$\hat{\mathbf{W}} = \mathbf{W} + \sum_{i=1}^{N} w_i \cdot \Delta \mathbf{W}_i$$
{{< /katex >}}


**Reference tuning-based composition** tackles the above limitations of linear arithmetic method by introducing gradient fusion and controllable sampling, but is requires retaining when incorporating different LoRAs or creating new masks, which results non-trivial computational costs.


<p align="center">
    <img src=./lora_comp.png> 
    <br>
    <em>(Left) Linear arithmetic composition. (Right) Reference tuning-based composition</em>
</p>


### Mixture-of-Experts

MoE is an effective method that allows scaling up the number of parameters while maintaining the computational cost of the model.

* Experts FFN Layers: MoE layer is composed of N separate feed-forward networks as the experts. This concept involves dividing the FFN layer of traditional transformers into N experts. These experts can be thought of as being responsible for specific tokens.

* Gating functions (Router): A function that determines the weights over the experts outputs. For the hidden representation h of input token, and the trainable embedding e of each a expert, the gate value a is obtained as follow:

{{< katex display=true >}}
\alpha(E_i) = \frac{\exp(h \cdot e_i)}{\sum_{j=0}^{N} \exp(h \cdot e_j)}
{{< /katex >}}

The output is a weighted sum of the outputs from the top-k experts, determined by the gated values.

{{< katex display=true >}}
O = h + \sum_{i=0}^{N} \alpha(E_i) \cdot E_i(h)
{{< /katex >}}

<p align="center">
    <img src=./moe.png> 
    <br>
    <em>Illustration of a Swith Transformer block.</em>
</p>

## Mixture of LoRA experts

### Motivations
1. Direct linear arithmetic composition reduced the generative power of the model, while normalized linear arithmetic composition retained the generative power of the model but lost its LORA character.
<p align="center">
    <img src=./motiv1_1.png align="center" width="40%">
    <img src=./motiv1_2.png align="center" width="40%">
    <figcaption align="center">
<p align="center">
    <img src=./motiv1_3.png width="700">
</p>
In the V&L domain, directly composing multiple trained LoRAs into the original embedding caused significant parameter variations and meaningless output, while normalization compromised their original characteristics. 
<br/>
In the NLP domain, composing four or more LoRAs within the FLAN-T5 model resulted in disordered output, and weight normalization across five datasets decreased the performance, suggesting adverse effects on the intrinsic qualities of the trained LoRAs.
<br/>
<br/>
2. Each layer of the trained LoRA represented a unique characteristic, which cumulatively defined the overall properties of the LoRA.
<p align="center">
    <img src=./motiv2_1.png align="center" width="40%">
    <img src=./motiv2_2.png align="center" width="53%">
    <figcaption align="center">
</p>
(Right: Observed that different layers of LoRA encode distinct features, such as dog coat color and facial features.,<br/>
left: When evaluated on a subset of datasets, there were significant differences in performance across the different layers of LoRA.) 
        
**So, The conjecture is that adjusting the characteristics by varying the layer-specific weights according to the desired domain objective will result in a more effective composition of trained LORAs.**
### Method

<p align="center">
    <img src=./Method1.png align="center" width="400">
</p>

<details>
    <summary>See related formulas</summary>
        <b>Symbols</b> <br/>
        input $x \in \mathbb{R} ^ {L \times d}$ <br/>
        L: sequence length <br/>
        d: dim of $x$ <br/>
        Multi attention layer : $$\mathcal{f}_{Attn} (\centerdot)$$ <br/>
        Feed forward neural network layer: $$\mathcal{f}_{FFN} (\centerdot)$$   <br/>
        LN: layer normalization <br/>
        Trained LORAs $$\Omega = \left\{ \Delta \Theta \right\}^N_{i=0}$$ <br/>
        learnable gating function $$\mathcal{G} (\centerdot)$$ <br/>
        The weight of the $i^{th}$ trained LorA $$\mathcal{G}_i (\centerdot)$$ <br/>
        Concatenation operation: $$\oplus$$ <br/>
        Learnable parameter $e \in \mathbb{R} ^ {N^2 \times L \times d}$ <br/>
        Learnable temperature scalar $\tau$ <br/>
        <br/>
        <b>Freezing part</b>
        $$x^\prime_{\theta} = x + \mathcal{f}_{Attn} (LN(x)|\theta)$$ <br/>
        $$\mathbf{F}_\theta (x) = x^\prime_{\theta} + \mathcal{f}_{Attn} (LN(x^\prime_{\theta})|\theta)$$ <br/>
        <br/>
        <b>LoRA part</b>
        $$x^\prime_{\Delta \Theta_i} = x + \mathcal{f}_{Attn} (LN(x)|\Delta \Theta_i)$$ <br/>
        The output of each LoRA $$\mathbf{E} _{\Delta \Theta_i} (x) = x^\prime_{\Delta \Theta_i} + \mathcal{f}_{FFN} (LN(x^\prime_{\Delta \Theta_i})|\Delta \Theta_i)$$ <br/>
        The output of all LoRA $$\mathbf{E}_\Omega (x) = Normalization(\mathbf{E}_{\Delta \Theta_0} (x) \oplus \ldots \oplus \mathbf{E}_{\Delta \Theta_{N-1}} (x)) \in \mathbb{R} ^ {N \times L \times d}$$ <br/>
        Flatten and dot product operation $$\epsilon = Flatten(\mathbf{E}_\Omega (x))^T \centerdot e,  \epsilon \in \mathbb{R} ^ N$$ <br/>
        Gate value for each LoRA $$\mathcal{G} (\epsilon_i) = \frac {exp(^{\epsilon_i} /_ \tau)} {\displaystyle\sum_{j=1}^N {exp(^{\epsilon_j} /_ \tau)}} $$ <br/>
        Final output of the gating function $${\tilde{\mathbf{E}}_\Omega (x)} = \displaystyle\sum_{i=0}^N {\mathcal{G} (\epsilon_i) \centerdot \mathbf{E} _{\Delta \Theta_i} (x)} , {\tilde{\mathbf{E}}_\Omega (x)} \in \mathbb{R} ^ {L \times d} $$ <br/>
        <b>Final output of Transformer block</b>
        $$\mathcal{O}(x) = {\mathbf{F}_\theta (x)} + {\tilde{\mathbf{E}}_\Omega(x)} $$ 
</details> 

### Training
The final loss function used in MoLE is as follows:
<p align="left">
    <img src=./training5.png width="200">
</p>
Alpha is a coefficient for weight balancing. 
<br/>

**Gating Balacing Loss**
<p align="center">
    <img src=./training1.png width="400">
</p>
As shown in Figure 5 (a), the average entropy of the distribution probabilities from the gating functions gradually decreases as training progresses. In Figure 5 (b), we can see a gating probability of 64% for LoRA β among the three LoRAs, indicating that the gating function tends to converge to a state where it assigns large weights to well-performing LoRAs in the early stages. This can result in a significantly larger impact from a few specific LoRAs compared to others, potentially leading to biased outcomes. <br/>
<br/>
To avoid this, the author created a gating balancing loss.<br/>
The gating balancing loss helps prevent bias by ensuring that the loss value decreases as the model becomes less biased. <br/>
<br/>
<p align="left">
    <img src=./training2.png width="200">
</p>
<details>
    <summary>See related Symbols</summary>
    M: The num of blocks where gating functions are placed <br/>
    N: num of LoRAs
</details>     
<br/>

**Domain-specific Loss**
<br/>
In V&L, Using a loss in CLIP(Radford et al,20221b) <br/>
<p align="left">
    <img src=./training3.png width="300">
</p>

In NLP, Using a loss in FLAN-T5(Chung et al,2022)
<p align="left">
    <img src=./training4.png width="200">
</p>

## Results

**On V&L Domain**
<br/>
- Setup)
  <br/>
  Base generator: DeamBooth(Ruiz et al., 2023) (built on Stable Diffusion V2.1)
  <br/>
  LoRA: combination of three separately trained LoRAs
  <br/>
  Image resolution: 512x512
  <br/>
  learning rate: 1e-5
  <br/>
  DDPM sampler (Ho et al., 2020) with 50 steps in each case
  <br/>
  Train 400 iterations for each required composition with batch size 2 and α as 0.5
  <br/>
- Metrics)
  <br/>
  Image alignment: Evaluate the visual similarity of generated images with individual composed concepts in the CLIP image feature space.
  <br/>
  Text alignment: Evaluate the text-image similarity of generated images with given text prompts in the CLIP feature space.
  <br/>
  For each composition, calculated the average scores among 200 generated images per prompt using 5 text prompts.
  <br/>
- Compared Baselines)
  <br/>
  - Normalized linear arithmetic composition
  - SVDiff (Han et al., 2023)
- Results)
  <br/>
<p align="center">
    <img src=./result1.png width="500">
</p>
        It demonstrates better performance compared to other models and shows outstanding results in other tasks as well.
<p align="center">
    <img src=./result2.png align="center" width="32%">
    <img src=./result3.png align="center" width="32%">
    <img src=./result4.png align="center" width="32%">
    <figcaption align="center">
</p>
  When viewing the generated images, it is evident that all specified subjects are accurately represented and maintained.
  <br/>
   <br/>
        
**On NLP Domain**
<br/>
- Setup)
  <br/>
  Base Model: Flan-T5 (Chung et al., 2022)
  <br/>
  LoRA: Several LoRAs based on FLAN datasets
  <br/>
  learning rate: 1e-5
  <br/>
  Train 800 iterations for each required composition with batch size 12 and α as 0.5.
  <br/>
- Compared Baselines)
  <br/>
  -  LoRAhub
  -  PEMs
- Results)
<p align="center">
    <img src=./result7.png align="center" width="48%">
    <img src=./result8.png align="center" width="48%">
    <figcaption align="center">
</p>
  It can be observed that MoLE demonstrates better performance in most tasks.
  
## Analyisis and Limitations












