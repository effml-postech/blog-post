---
type: docs
bookToc: True
weight: 1
---

# MIXTURE OF LORA EXPERTS

LoRA is a methodology for effective fine-tuning large-scale pretrained models. LoRA is characterized by its ease of applying tuned results to existing models. This property encouragees research into synthesizing multiple trained LoRAs to achieve enhanced performance across various tasks such as linear arithmetic composition and reference tuning-based composition. However, combining these trained LoRAs poses significant two challenges:

1. **Linear arithmetic composition can diminish the capabilities of the original pre-trained models and the unique characteristics of the individually trained LoRAs, potentially leading to suboptimal results.**

2. **Reference tuning-based composition is limited in adaptability and incurs substantial computational costs, as it necessitates retraining a large model.**

So, we can ask following question:

<p align="center">
_How can multiple trained LoRAs be composed dynamically and efficiently, preserving all their individual characteristics, without the need for retraining?_
</p>

To address this question, Mixture of LoRA Experts (MoLE) presents a new method for achieving the optimal combination of LoRAs for specific tasks. MoLE considers indivisual LoRA as an expert and determines the weights applied to LoRAs at each layer through a gate function.


<p align="center">
    <img src="./mole.png">
    <br>
    <em>Workflow of Mixture of LoRA Experts (MoLE)</em>
</p>

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

The common solution to further improve the performance of LoRA across various tasks is to compose multiple trained LoRAs. Research on LoRA composition can be broadly categorized into the following two methodologies.

* _**Linear arithmetic composition.**_ It is a method of directly adding multiple LoRAs. This approach is simple and has been effective in the NLP and Vision-Language domain, but it can result in the loss of pre-trained model's generative capabilities or the individual characteristics of each LoRA.

{{< katex display=true >}}
\hat{\mathbf{W}} = \mathbf{W} + \sum_{i=1}^{N} w_i \cdot \Delta \mathbf{W}_i
{{< /katex >}}


* _**Reference tuning-based composition**_ tackles the above limitations of linear arithmetic method by introducing gradient fusion and controllable sampling, but is requires retaining when incorporating different LoRAs or creating new masks, which results non-trivial computational costs.


<p align="center">
    <img src=./lora_comp.png> 
    <br>
    <em>(Left) Linear arithmetic composition. (Right) Reference tuning-based composition</em>
</p>


### Mixture-of-Experts

MoE is an effective method that allows scaling up the number of parameters while maintaining the computational cost of the model.

<p align="center">
    <img src=./moe.png> 
    <br>
    <em>Illustration of a Swith Transformer block.</em>
</p>

* Experts FFN Layers: MoE layer is composed of N separate feed-forward networks as the experts. This concept involves dividing the FFN layer of traditional transformers into N experts. These experts can be thought of as being responsible for specific tokens.

* Gating functions (Router): A function that determines the weights over the experts outputs. For the hidden representation h of input token, and the trainable embedding e of each a expert, the gate value a is obtained as follow:

{{< katex display=true >}}
\alpha(E_i) = \frac{\exp(h \cdot e_i)}{\sum_{j=0}^{N} \exp(h \cdot e_j)}
{{< /katex >}}

The output is a weighted sum of the outputs from the top-k experts, determined by the gated values.

{{< katex display=true >}}
O = h + \sum_{i=0}^{N} \alpha(E_i) \cdot E_i(h)
{{< /katex >}}

## Mixture of LoRA experts

### Observations
1. Direct linear arithmetic composition reduced the generative power of the model, while normalized linear arithmetic composition retained the generative power of the model but lost its LORA character.


<p align="center">
    <img src=./motiv1_1.png align="center" height="150">
    <img src=./motiv1_2.png align="center" height="150">
    <br>
    <em>(Left) Result of linear arithmetic composision. (Right) Result of nomalized linear arithmetic composision.</em>
</p>

<p align="center">
    <img src=./motiv1_3.png width="40%">
    <br>
    <em>Experiment in the NLP domain. NLA denotes normalized linear arithmetic composision </em>
</p>

In the V&L domain, directly composing multiple trained LoRAs into the original embedding caused significant parameter variations and meaningless output, while normalization compromised their original characteristics.  In the NLP domain, composing four or more LoRAs within the FLAN-T5 model resulted in disordered output, and weight normalization across five datasets decreased the performance, suggesting adverse effects on the intrinsic qualities of the trained LoRAs.

2. Each layer of the trained LoRA represented a unique characteristic, which cumulatively defined the overall properties of the LoRA.


<p align="center">
    <img src=./motiv2_1.png align="center" height=150>
    <img src=./motiv2_2.png align="center" height=150">
    <figcaption align="center">
    <br>
    <em>(Right) Observed that different layers of LoRA encode distinct features, such as dog coat color and facial features. (Left) When evaluated on a subset of datasets, there were significant differences in performance across the different layers of LoRA.) </em>
</p>
        
**So, The conjecture is that adjusting the characteristics by varying the layer-specific weights according to the desired domain objective will result in a more effective composition of trained LORAs.**

### Method

<p align="center">
    <img src=./Method1.png align="center" width="70%">
    <br>
    <em>Illustration of proposed MOLE. MOLE employs a learnable gating function that utilizes the outputs of multiple LoRAs at each layer to determine composition weights.</em>
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
The training loss function used in MoLE is as follows:
<p align="centor">
    <img src=./training5.png width="200">
</p>

{{< katex display=true >}}
$$\mathcal{L} = \mathcal{L}_{D} + \alpha \mathcal{L}_{balance}$$
{{< /katex >}}


Alpha is a coefficient for weight balancing. 

**Gating Balacing Loss**
<p align="center">
    <img src=./training1.png width="400">
</p>
As shown in Figure 5 (a), the average entropy of the distribution probabilities from the gating functions gradually decreases as training progresses. In Figure 5 (b), we can see a gating probability of 64% for LoRA β among the three LoRAs, indicating that the gating function tends to converge to a state where it assigns large weights to well-performing LoRAs in the early stages. This can result in a significantly larger impact from a few specific LoRAs compared to others, potentially leading to biased outcomes. <br/>
<br/>
To avoid this, the author created a gating balancing loss.<br/>
The gating balancing loss helps prevent bias by ensuring that the loss value decreases as the model becomes less biased. <br/>
<br/>
<p align="centor">
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
<p align="centor">
    <img src=./training3.png width="300">
</p>

In NLP, Using a loss in FLAN-T5(Chung et al,2022)
<p align="centor">
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
  
## Analyisis 

### 1. Gating balancing loss works!
Gating balancing loss function mitigates the reduction in entropy rates within gating functions, and enhance the performance.

<p align="center">
    <img src=./anal1-2.png> 
    <br>
    <em>Experimental results on gating balance of MOLE. NLA denotes normalized linear arithmetic composition</em>
</p>

### 2. MoLE is even better than SOTA multi-concept generation methods.
MoLE outperforms two multi-concept generation algorithms (Custom, Textual Inversion), both of which emphasize full-parameter training for enhanced results.

<p align="center">
    <img src=./anal2.png> 
    <br>
    <em>Text-alignment and image-alignment results for multiple LoRA experts composition in CLIP feature space. SOTA full-parameter training methods are highlighted by pink boxes</em>
</p>

### 3. Scale to a larger number of LoRAs.
MOLE demonstrated optimal performance across varying numbers of LoRA, notably surpassing LoRAHub with larger LoRA counts of 48 and 128. However, all methods, including MOLE, showed performance declines with an extremely large number of LoRA 

<p align="center">
    <img src=./anal3.png> 
    <br>
    <em>NLP domain experimental results on the impact of exploring expand expert numbers on model performance. The result is the average EM on the Big-Bench Hard (BBH) dataset.</em>
</p>

### 4. Coarse gating vs. fine gating
Among  matrix-wise, layer-wise, block-wise, and network-wise MoLEs, intermediate granularities, b-MoLE and l-MoLE, achieved the highest performance.

<p align="center">
    <img src=./anal4.png> 
    <br>
    <em>Coarse-to-fine gating comparison</em>
</p>

### 5. Flexibility of MoLE.
MoLE not only achieves effective LoRA composition but also retains the characteristics of individual LoRA. It can generate images that closely resemble the original features of the LoRA experts

<p align="center">
    <img src=./anal5.png> 
    <br>
    <em>(Left) Linear arithmetic composition. (Right) Reference tuning-based composition</em>
</p>

### 6. Hierarchical control analysis
MOLE adaptively assigns weights to different LoRA experts across various layers, resulting in finer-grained weight combinations that yield superior results.

<p align="center"
    <img src=./anal6.png> 
    <br>
    <em>Visualization of the weights (%) predicted by each gating function (horizontal axis) for LoRA experts (vertical axis) during inference. The top row corresponds to experiments in the NLP domain, while the bottom row pertains to experiments in the V&L domain.</em>
</p>
   
## Discussion and Limitations
**Limitations**
1. LoRA scale <br/>

When the number of LoRAs increases to a very large value (e.g., 128), the performance of all LoRA composition methods, including MOLE, tends to decrease despite MOLE's superior performance. This indicates that MOLE still faces challenges with large-scale LoRA composition and emphasizes the need for better approaches to handle it effectively. 

2. Parameter <br/>

The learnable parameter 𝑒 used in MoLE has dimensions of $N^2 \times L \times D$. As the number of LoRAs increases, the number of parameters grows quadratically, resulting in a substantial increase. Additionally, since e exists for each transformer block, the number of parameters added by 𝑒 is considerable. This can be seen as a drawback of MoLE.
<br/>

**Discussion**

<How to address MoLE's limitations at LoRA scale> <br/>
Currently, MoLE's performance decreases when the number of LoRAs exceeds a certain threshold. By reducing the number of LoRAs to below this threshold with minimal loss, performance could be improved. Assuming there is a large number of LoRAs, there will likely be many LoRAs for similar tasks. Given this, we believe that clustering to derive representative LoRAs for similar tasks and using only the representative LoRAs instead of all similar task LoRAs could overcome MoLE's limitations.















