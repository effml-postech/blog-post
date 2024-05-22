---
type: docs
bookToc: True
weight: 1
---

# **ViTAR: Vision Transformer with Any Resolution**
*Authors: Qihang Fan, Quanzeng You, Xiaotian Han, Yongfei Liu, Yunzhe Tao, Huaibo Huang, Ran He, Hongxia Yang*

*Posted by Jungwon Lee, Minsang Seok*

----------

## What is Vision Transformer?

Vision Transformer (ViT) is an innovative approach to computer vision that leverages the principles of the Transformer architecture, which was originally designed for natural language processing tasks. ViT has recently emerged as a competitive alternative to Convolutional Neural Networks (CNNs) that are currently state-of-the-art in different image recognition computer vision tasks.

<p align="center">
  <img src="./ViT.png" alt="." width="700" height="400" > 
</p>

Vision Transformer architecture consists of a series of Transformer blocks, each containing a multi-head self-attention layer and a feed-forward layer. This structure allows ViT to capture complex relationships within an image more effectively than traditional convolutional layers.

### Key Components of ViT
The key coomponents of ViT are described below:

#### A. Patch Embedding
- Instead of processing the entire image as a whole, ViT divides the input image into fixed-size patches (e.g., 16x16 pixels).
Each patch is then flattened into a single vector, essentially treating each patch as a "token" similar to how words are treated in text processing. These flattened patch vectors are linearly projected to a desired embedding dimension. This projection helps in transforming the patches into a suitable format for the Transformer model.

#### B. Positional Encoding
- Since Transformers are permutation-invariant and do not inherently understand the spatial relationships between patches, positional encodings are added to the patch embeddings. These encodings provide information about the position of each patch in the original image.

#### C. Self Attention
- The self-attention layer calculates attention weights for each pixel in the image based on its relationship with all other pixels.

- For each input vector X, three new vectors are created through learned linear transformations: Query (Q), Key (K), and Value (V), where {{< katex >}}W_{Q}, W_{K}, W_{V}{{< /katex >}} are learnd weight matrices.

{{< katex display=true >}}
Q = XW_Q, K=XW_K, V=XW_V
{{< /katex >}}

- The attention score for each pair of input vectors is calculated using the dot product of their Query and Key vectors:

{{< katex display=true >}}
Attention Score = Q K^T
{{< /katex >}}

- These scores indicate how much focus the model should place on one part of the input when considering another part.

{{< katex display=true >}}
Attention Output = softmax(\frac{Q \dot K^T}{\sqrt{d_k}}V)
{{< /katex >}}



- The attention scores are scaled by the square root of the dimensionality of the Key vectors to prevent excessively large values that could destabilize training. The scaled attention scores are passed through a softmax function to obtain the attention weights. This ensures that the weights are normalized (summing to one) and highlight the relative importance of each input vector. Each input vector is then updated by computing a weighted sum of the Value vectors, using the attention weights.

#### C. Multi-Head Self Attention (MHSA)
- The multi-head attention extends self-attention mechanism by allowing the model to attend to different parts of the input sequence simultaneously. Each "head" in the multi-head attention mechanism can capture different features, leading to a richer and more nuanced representation of the image.

#### D. Feedforward Neural Networks:
- Each self-attention layer is followed by a feedforward neural network that further processes the information.
These networks consist of fully connected layers and typically include activation functions and normalization.

If interested in more details about ViT, please refer to the following paper. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"

----------

## Challenge: Multi-Resolution ViT Modeling
 Shortcoming of ViT is revealed when receiving multi-resolution images as input. There are limits to its application in actual use environments because ViT cannot process images of various resolutions well.
 
 The most common method used to address this problem is to apply interpolation to positional encoding before feeding it into the ViT. This approach allows for some compensation of positional information even when the input resolution changes. However, this method has shown significant performance degradation in image classification tasks.
 
 Recently, ResFormer proposed adding depth-wise convolution to the existing positional encoding method when performing global-local positional embedding, enabling it to work well even with unseen resolutions. (Chu et al., 2023; Tian et al., 2023).

<p align="center">
  <img src="./ResFormer_pe.png" alt="." width="600" height="350" > 
</p>

 However, ResFormer has three drawbacks.
 - Shows high performance only in a relatively small range of resolutions (Degradation significantly when resolution is greater than 892)
 - It cannot be used with self-supervised learning methods like masked auto-encoding (MAE).
 - Computation cost increases as input resolution increases, which has a negative impact on the training and inference process.

----------

## ViTAR: Vision Transformer with Any Resolution
<p align="center">
  <img src="./ViTAR_overall.png" alt="." width="600" height="200" > 
</p>

To address this issue, ViTAR introduces two key innovations. 
- 1. Adaptive Token Merger : A novel module for dynamic resolution adjustment, designed with a single Transformer block to achieve highly efficient incremental token integration.
- 2. Fuzzy positional encoding : A novel positional encoding to ensure consistent positional awareness across multiple resolutions, thereby preventing overfitting to any specific training resolution.
  
----------

### 1. Adaptive Token Merger (ATM Module)

<p align="center">
  <img src="./ATM.png" alt="." width="500" height="300" > 
</p>    

    
Adaptive Token Merger (ATM) module is designed to efficiently process and merge tokens of different resolutions in a neural network using a simple structure that includes GridAttention and FeedForward network (FFN). ATM Module takes tokens processed through patch embedding as input. ATM Module specially processes the inputs of different resolutions M times to reduce them to the same preset size {{< katex >}}G_{h} \times G_{w}{{< /katex >}} before fed into the MHSA.

<p align="center">
  <img src="./grid_attention.png" alt="." width="600" height="300" > 
</p>

The detailed process for ATM is as follows:  

 First, ATM divides the tokens of shape {{< katex >}}(H \times W){{< /katex >}} into a grid of size {{< katex >}}G_{th} \times G_{tw}{{< /katex >}}. 
 
 For simplicity, we'll use above Figure as an example. 
 In the figure, we can see {{< katex >}}H=4{{< /katex >}}, {{< katex >}}W=4{{< /katex >}}, {{< katex >}}G_{th}=2{{< /katex >}}, and {{< katex >}}G_{tw}=2{{< /katex >}}.(We assume that H is divisible by {{< katex >}}G_{th}=2{{< /katex >}} and W is divisible by {{< katex >}}G_{tw}=2{{< /katex >}}. The number of tokens in each grid would then be {{< katex >}}H/G_{th} × W/G_{tw}{{< /katex >}}, which is 2x2.

 Within each grid, the module performs a special operation called Grid Attention.

 #### GridAttention 
For a specific grid, we suppose its tokens are denoted as {{< katex >}}{x_{ij}}{{< /katex >}}, where {{< katex >}}0 \leq i < H/G_{th}{{< /katex >}} and {{< katex >}}0 \leq j < W/G_{tw}{{< /katex >}}. 

- Average Pooling: First, it averages the tokens within a grid to create a mean token.
- Cross-Attention: Using this mean token as the Query, and all the grid tokens as Key and Value, it applies cross-attention to merge all tokens in the grid into a single token.

{{< katex display=true >}}
x_{avg} = AvgPool(\{x_{ij}\})
{{< /katex >}}

{{< katex display=true >}}
GridAttn(\{x_{ij}\}) = x_{avg} + Attn(x_{avg}, \{x_{ij}\}, \{x_{ij}\})
{{< /katex >}}

 After passing through GridAttention, the fused token is fed into a standard Feed-Forward Network to complete channel fusion, thereby completing one iteration of merging token. GridAttention and FFN undergo multiple iterations and all iterations share the same weights. 
 
  During these iterations, we gradually decrease the value of {{< katex >}}(G_{th} , G_{tw}){{< /katex >}}, until {{< katex >}}G_{th} = G_{h}{{< /katex >}} and {{< katex >}}G_{tw} = G_{w}{{< /katex >}}. (typically set {{< katex >}}G_h = G_w = 14{{< /katex >}}, in standard ViT)

 This iteration process effectively reduces the number of tokens even when the resolution of the image is large, and with enough iterations, this size can be reduced effectively. This has the advantage of being computationally efficient because when performing subsequent MHSA calculations, we always use the same size tokens as input, regardless of resolution.

----------

For Ablation study, ViTAR-S Model is used to compare with AvgPool which is another token fusion method. The results of the comparison demonstrate that ATM significantly improves the model's performance and resolution adaptability. Specifically, at a resolution of 4032, our proposed ATM achieves a 7.6\% increase in accuracy compared with the baseline.

<p align="center">
  <img src="./result_ATM.png" alt="." width="600" height="250" > 
</p>

----------

### 2. Fuzzy Positional Encoding (FPE)
 Existing ViT Models generally use learnable positional encoding or sin-cos positional encoding. However, they do not have the ability to handle various input resolutions because these methods are sensitive to input resolution. In response to this, ResFormer attempted to solve this problem through convolution-based positional embedding.

 However, convolution-based positional embedding is not suitable for use in self-supervised learning such as masked auto-encoding (MAE). This is because the method can extract and utilize the complete spatial feature only if it has all adjacent patches, but in the case of MAE, some of the image patches are masked. This makes it difficult for the model to conduct large-scale learning.

 Fuzzy Positional Encoding(FPE) differs from the previously mentioned methods. It enhances the model's resolution robustness without introducing specific spatial structures like convolutions. Therefore, it can be applied to self-supervised learning frameworks. This property enables ViTAR to be applied to large-scale, unlabeled training sets for training, aiming to obtain a more powerful vision foundation model.

<p align="center">
  <img src="./FPE.png" alt="." width="500" height="300" > 
</p>

 Initially, the learnable positional embedding is randomly initialized and used as the model's positional embedding. At this time, FPE provides only fuzzy positional information and experiences changes within a certain range. Specifically, assuming that the exact coordinates of the target token are (i, j), the fuzzy positional information is (i + s1, j + s2). s1 and s2 satisfy -0.5 ≤ s1, s2 ≤ 0.5 and follows uniform distribution.

 During training, randomly generated coordinate offsets are added to the reference coordinates during the training process, and grid samples for learnable location embeddings are performed based on the newly generated coordinates to generate fuzzy location encoding.

 In case of inference, precise positional encoding is used instead of FPE. When there is a change in input resolution, interpolation is performed on learnable positional embedding. This has strong positional resilience because it was somehow seen and used in the FPE used in the training phase.

----------

 To compare the impact of different positional encodings on the model’s resolution generalization ability, several positional encoding methods were used. This includes commonly used sin-cos absolute position encoding (APE), conditional position encoding (CPE), global-local positional encoding (GLPE) in ResFormer, Relative Positional Bias (RPB) in Swin, and FPE. Note that only APE and FPE are compatible with the MAE framework.ViTAR-S is used for experiments without MAE, and ViTAR-M is used for experiments with MAE. As a result, FPE exhibits a significantly pronounced advantage in resolution generalization capability. Additionally, under the MAE self-supervised learning framework, FPE also demonstrates superior performance relative to APE.
<p align="center">
  <img src="./result_FPE.png" alt="." width="400" height="300" > 
</p>

----------

## ViTAR shows superior performance with any resolution


### Image Classification

ViTAR is trained on ImageNet-1K form scratch and it demonstrates excellent classification accuracy across a considerable range of resolutions. Especially, when the resolution of the input image exceeds 2240, ViTAR is capable of inference at lower computational cost. In contrast, traditional ViT architectures (DeiT and ResFormer) cannot perform high resolution inference due to computational resource limitations.

<p align="center">
  <img src="./result_image_classification.png" alt="." width="600" height="600" > 
</p>

As can be seen in the pareto frontier figure, ViTAR has high performance for various resolution images and can also be used for high resolution images of 2240 or higher.

<p align="center">
  <img src="./result1.png" alt="." width="500" height="300" > 
</p>

### Object Detection
For object detection, COCO dataset is used ATM iterates only once because it does not utilize the multi-resolution training strategy in this experiment. If {{< katex >}}\frac{H}{G_{th}}{{< /katex >}} and {{< katex >}}\frac{w}{G_{tw}}{{< /katex >}} in ATM are fixed to 1, the results indicate that ViTAR achieves performance in both object detection and instance segmentation. And if setting {{< katex >}}\frac{H}{G_{th}}{{< /katex >}} and {{< katex >}}\frac{w}{G_{tw}}{{< /katex >}} to 2 in ATM, ATM module reduces approximately 50\% of the computational cost while maintaining high precision in dense predictions, demonstrating its effectiveness.

<p align="center">
  <img src="./result_object_detection.png" alt="." width="800" height="350" > 
</p>

----------

## Discussion

### Applicability to Diffusion Models
- It is currently challenging to generate images of various resolutions with generative models like Diffusion Models. Additionally, many diffusion models with ViT structures have been proposed recently (e.g. DiT, PixArt-α, Sora). Can the proposed method be applied to Diffusion Models as well? However, one consideration for applying it to diffusion models is how to effectively upscale the reduced size obtained through Grid Attention to ensure that the input and output sizes are the same.

### Applicability to Large Language Models (LLMs)
- In LLMs, when receiving long context as input, positional embeddings are sometimes added using interpolation like this case. Would applying Fuzzy Positional Embedding (FPE) help handle long context inputs better? Or, just like training a network on low-resolution images to perform well on high-resolution images, can a network trained on short context in LLM maintain good performance on long context input?

### Can Grid Attention Replace Convolution?
- The operation of GridAttention is quite similar to the process performed by kernels in Convolution when calculating each grid. However, ATM maintains parameter efficiency by sharing weights. We expect that applying GridAttention to existing CNN structures (e.g., VGG, ResNet) will allow us to design more efficient architectures.

