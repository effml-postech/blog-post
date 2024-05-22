---
type: docs
bookToc: True
weight: 1
---
# **Evolutionary Optimization of Model Merging Recipes**
*Authors: Takuya Akiba, Makoto Shing, Yujin Tang, Qi Sun, David Ha., arXiv 2024*

*Posted by Jaehyun Pahk, Youngkil Song*

## Background
### Evolutionary Algorithm
The evolutionary algorithm is a type of black box optimization that can achieve the desired optimization without directly computing the gradient ▽f(x) or the Hessian ▽²f(x) of the function.
Evolutionary algorithms are typically composed of the following elements:
- **Population**: A set of possible solutions to the problem, where each individual represents a candidate solution.
- **Fitness Function**: A function that evaluates the quality of each individual. The fitness function reflects the objective of the problem, with higher values indicating better solutions.
- **Selection**: The process of selecting individuals to be passed on to the next generation. Individuals with higher fitness are more likely to be selected.
- **Termination Condition**: The criteria for ending the algorithm. This is usually based on a maximum number of generations, achieving a target fitness level, time limits, or other benchmarks.

#### Simple Evolutionary Strategy (Simple ES)
1. Sample a set of solutions from a Normal distribution with mean 𝜇 and a fixed standard deviation 𝜎.
2. Evaluate the fitness of each solution using the fitness function.
3. Set 𝜇 to the best solution in the population, and sample the next generation of solutions around this new mean.
4. Repeat the above processes.

#### Covariance-Matrix Adaptation Evolution Strategy (CMA-ES)
CMA-ES is an evolutionary strategy that can dynamically adjust the search range for solutions.
CMA-ES finds the global optimum effectively even in high-dimensional problems by adaptively updating the covariance matrix of the multivariate normal distribution used for sampling a set of solutions, thereby adjusting the search direction and range. Fig. 1 from [article](https://blog.otoro.net/2017/10/29/visual-evolution-strategies) shows a simple CMA-ES simulation in a 2D space.

<p align="center">
  <img src="./figures/CMA-ES.gif" alt="." width="500" height="300"></br>
  Figure 1. A 2D Simulation of CMA-ES Algorithm.
</p>

### How to leverage the strengths of multiple pre-trained models?
#### Fine-Tuning

__Fine-tuning__ involves taking a pre-trained model and further training it on a specific dataset to optimize its performance for a particular task.

- **Base Model** → **Task-Specific Model**
  - Example: Fine-tuning a general language model on a medical dataset to create a medical chatbot.

**Advantages**
- **Specialized Performance**: Highly effective for optimizing models for specific tasks.
- **Efficiency**: Requires less computational power and time compared to training from scratch.
- **Flexibility**: Can be applied to any pre-trained model to fine-tune it for different tasks.

**Disadvantages**
- **Overfitting**: Risk of overfitting to the small dataset used for fine-tuning.
- **Limited Generalization**: May not perform well on tasks outside the fine-tuned domain.
- **Dependent on Pre-trained Model**: The performance heavily relies on the quality of the pre-trained model.


#### Model Merging

__Model merging__ involves combining multiple pre-trained models into a single model by integrating their weights and architectures to leverage their collective strengths.

- **Model A** (e.g., Japanese LLM) + **Model B** (e.g., Math LLM) → **Merged Model**
  - Example: Combining a Japanese language model with a mathematical reasoning model to create a Japanese math reasoning model.

**Advantages**
- **Cost-Effective**: Does not require additional training, making it computationally efficient.
- **Enhanced Capabilities**: Can combine strengths from different models, potentially handling a broader range of tasks.
- **Cross-Domain Application**: Effective in creating models that perform well across different domains (e.g., language and vision).

**Disadvantages**
- **Complexity**: The merging process can be complex and requires careful selection and comparison of models.
- **Black-Box Nature**: May be seen as less interpretable since it relies on heuristic methods for weight integration.
- **Potential for Suboptimal Performance**: If not done correctly, merged models may not achieve the desired performance improvements.

<p align="center">
  <img src="./figures/merging model figure.png" alt="." width="500" height="300" > 
</p>

<p align="center">
  Figure 2. Example of Model Merging.
</p>

Fig. 2 from ([Xu et al., CVPR 2024](https://arxiv.org/abs/2403.01753)) illustrates an example of model merging. This approach involves pair-wise comparison of weights from two pre-trained models and merging the most similar weights together.

### Merging Language Models
Research on applying model merging to language models is actively progressing, with a large number of capable merged models being developed by the community. As a result, most of the top models on [the Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) are increasingly dominated by merged models produced by language model enthusiasts.

**Mergekit**
[github](https://github.com/arcee-ai/mergekit)

__Mergekit__ is a toolkit that provides various popular recipes for merging language models. It includes both simple methods like linear and spherical interpolation as well as more advanced techniques.

**Advanced Merging Methods**

- **Task Arithmetic**: Involves creating task vectors by subtracting the weights of a pre-trained model from a fine-tuned model and then manipulating these vectors to steer the merged model’s behavior.
- **TIES-Merging**: Addresses parameter interference by resetting minimal parameter changes, resolving sign conflicts, and merging only aligned parameters. This approach aims to mitigate information loss during the merging process.
- **DARE (Differentiable Adaptation and Regularization of Ensembles)**: Amplifies significant differences between models while zeroing out small differences. Often used in conjunction with Task Arithmetic or TIES-Merging to improve merging performance.




## Contributions
- **Automated Model Composition**: Developed an evolutionary method to automatically discover optimal combinations of diverse open-source models, creating powerful models without extensive training data or compute resources.
- **Cross-Domain Merging**: Demonstrated the ability to merge models from different domains (e.g., language and Math, language and Vision) to achieve enhanced capabilities beyond conventional design.
- **State-of-the-Art Performance**: Achieved state-of-the-art results with automatically generated models, including a Japanese LLM with Math reasoning capability and a culturally-aware Japanese VLM.
- **High Efficiency and Surprising Generalizability**: Showed that a 7B parameter model outperformed some 70B parameter models, highlighting efficiency and generalization capabilities.
- **Culturally-Aware VLM**: Produced a Japanese VLM that excels in handling Japanese culture-specific content, achieving top results on relevant benchmarks.

## Method
### Explanation of overall method
The goal of this paper is to develop a unified framework that can automatically generate a merged model from a set of foundation models, ensuring that the merged model outperforms an any single model in the collection. In this paper, an evolutionary algorithm was applied to reduce the complexity of the model merge process. The model merge was applied independently and also sequentially in both parameter space and the data flow space.
 
### Merging in the Parameter Space (PS)
The model merge in the parameter space can be summarized as a weighted average of the model parameters. In this paper, the fitness of each foundation model for a specific task is determined using the task vector of each foundation model, and then merging configuration parameters for combining the parameters of the candidate models are estimated based on those fitness values. Specifically, this paper enhances TIES-Merging with DARE, allowing for more granular, layer-wise (input/output embedding layers or transformer blocks) merging. Fig. 3 of [Sakana.ai](https://sakana.ai/evolutionary-model-merge/) shows an overview of the PS merging.

<p align="center">
  <img src="./figures/PS.gif" alt="." width="500" height="300"></br>
  Figure 3. Model Merging in the Parameter Space.  
</p>

### Merging in the Data Flow Space (DFS)
In DFS, the proposed framework discovers the best combinations of the layers of different models to form a new model, without changing the model parameters. In other words, the goal of merging in the DFS is to find the optimal inference path across the multiple models. For example, after the i-th layer in model A, a token may be directed to the j-th layer in model B. Fig. 4 of [Sakana.ai](https://sakana.ai/evolutionary-model-merge/) shows an overview of the DFS merging.

<p align="center">
  <img src="./figures/DFS.gif" alt="." width="500" height="300"></br>
  Figure 4. Model Merging in the Data Flow Space.  
</p>

Please note that the search space in the data flow space (DFS) is very large. Assuming the total number of layers across all models is $M$ and the lengh of the inference path is $T$, then the size of the search space is $M^T$. This astronomically large search space leads to a challenge for a evolutionary search algorithm, even with a modest configuration of $M=64$ and $T=60$. </br>
To address this issue, this paper exploits the result of preliminary studies that certain layer arrangements, particularly repetitive or permuted sequences from earlier in the model, can adversely affect performance. Specifically, this paper layout all the layers only in sequential order (i.e., all layers in the $i$-th model followed by those in the $i+1$-th model) and repeat them $r$ times, therefore the size of the search space can be reduced to $2^{M \times r}$. The authors use indicator array $\mathcal{I} \in \mathbb{R}^{M \times r}$ to represent which layers are included and excluded. </br>
However, in the above setting, a layer may face an input whose distribution is different from what it is used to (from its original model), leading to unexpected outputs. They just apply scaling the input based on the scaling matrix $W \in \mathbb{R}^{M \times M}$, which is also optimized by the evolutionary search together with the indicator array $\mathcal{I}$.

### Merging in Both Spaces
Model merging in the parameter space (PS) and data flow space (DFS) can be applied orthogonally to boost the performance of the merged model. Specifically, in this paper, model merging is first applied in the PS to generate several merged models, which are then put back to the collection of models. The expanded collection is subsequently used for merging in the DFS. Fig. 5 of [Sakana.ai](https://sakana.ai/evolutionary-model-merge/) shows an overview of the overall method.

<p align="center">
  <img src="./figures/Overall.gif" alt="." width="500" height="300"></br>
  Figure 5. Overall Method.  
</p>

## Experiments
The experiments in the paper focus on applying the proposed evolutionary model merging approach to create advanced models in two primary areas: Japanese LLMs with Math reasoning capabilities and culturally-aware Japanese Vision-Language Models (VLMs).
### Evolving Japanese Math LLM
### Setup
- **Source Models**: 
  - Japanese LLM: shisa-gamma-7b-v1
  - Math LLMs: WizardMath-7B-V1.1, Abel-7B-002
- **Dataset**: 
  - Training: 1069 translated samples from GSM8k test set
  - Testing: 250 samples from the Japanese test set of MGSM
- **Evaluation**: 
  - Accuracy measured by the correctness of the numerical value and reasoning text in Japanese.
  - Used fasttext for language detection and greedy sampling for generation.
- **Optimization**: 
  - Parameter Space (PS): Used CMA-ES algorithm implemented in Optuna for optimization.
  - Data Flow Space (DFS): Limited to two models with a budget of T = 192 steps, using CMA-ES in EvoJAX for optimization.

<p align="center">
  <img src="./figures/4_1 experiment table1.png" alt="." width="800" height="400" > 
</p>

<p align="center">
  Table 1. Performance Comparison of the LLMs on both MGSM-JA and JP-LMEH benchmarks.
</p>

<p align="center">
  <img src="./figures/4_1 experiment table2.png" alt="." width="800" height="400" > 
</p>

<p align="center">
  Table 2. Breakdown of JP-LMEH Scores for Japanese Language Proficiency.
</p>

### Results
- **Performance**: 
  - Merged models in PS and DFS showed substantial performance improvements, with the hybrid model (PS+DFS) achieving the highest accuracy.
  - The PS merged model (Model 4) scored 52.0 on MGSM-JA, while the hybrid model (Model 6) scored 55.2 in Table 1.
  - The PS merged model (Model 4) scored 70.5 on JP-LMEH, while the hybrid model (Model 6) scored 66.2 in Table 1.
- **Analysis**:
  - While they validate the effectiveness of the evolutionary model merging approach, it is challenging to determine the superiority of the PS, DFS, and merged model methods. Notably, as shown in Table 2, the PS method exhibits the highest accuracy on the JP-LMEH benchmark.

### Evolving Japanese VLM
### Setup
- **Source Models**: 
  - Japanese LLM: shisa-gamma-7b-v1
  - VLM: LLaVA-1.6-Mistral-7B
- **Dataset**: 
  - JA-VG-VQA-500: 500 samples from the Japanese Visual Genome VQA dataset.
  - JA-VLM-Bench-In-the-Wild: 42 images with 50 questions, focusing on Japanese cultural elements.
- **Evaluation**: 
  - Baselines: LLaVA-1.6-Mistral-7B and Japanese Stable VLM.
  - ROUGE-L score used for evaluation, with non-Japanese responses replaced by empty texts.

<p align="center">
  <img src="./figures/4_2 experiment table1.png" alt="." width="500" height="250" > 
</p>

<p align="center">
  Table 3. Performance Comparison of the VLMs.
</p>

### Results
- **Performance**: 
  - As shown in Table 3, the merged VLM outperformed baselines on both benchmarks, scoring 19.7 on JA-VG-VQA-500 and 51.2 on JA-VLM-Bench-In-the-Wild.
- **Qualitative Analysis**: 
  - The merged VLM demonstrated superior handling of Japanese cultural content, providing more detailed and accurate responses compared to baseline models.


## Conclusion
The evolutionary model merging method, which operates in both parameter space (PS) and data flow space (DFS), challenges the conventional paradigm of expensive model development, offering a more efficient alternative that can produce competitive models without relying on gradient-based training. This paper is significant in that it attempts to automate the merging of LLM foudation models based on an evolutionary algorithm, which previously relied on human intuition.

__However__, There are limitations to discuss.
### Limitation
- **Contradiction in Automation**: Although the method aims to automate the merging process by removing prior information and heuristic components, it still relies on experimental experience for design and learning methods, which is contradictory to the initial motivation.
- **Complexity and Efficiency of the Model**: By applying the PS merging and the DFS merging orthogonally and sequentially, the temporal and spatial complexity for model merging seems to have increased significantly.
- **Effectiveness of applying both merging methods**: We also observe in the experiments that the performance of the merged model is better when PS merging is applied alone than when both PS merging and DFS merging are applied, which raises the question of whether applying both merging methods together is worth the increased complexity of merging.
- **Limited Language Scope**: The paper presents experimental results only for Japanese (JP), which is not convincing for demonstrating the method's superiority in non-English languages.
- **Lack of Comparative Experiments**: There is a lack of experimental comparison with various other methods that use pre-trained models for multiple tasks such as fine-tuning.

### Future Work

<p align="center">
  <img src="./figures/post dfs merging.png" alt="." width="400" height="300" > 
</p>
<p align="center">
  Figure 6. Evolved Configurations for DFS Merging of models.
</p>

Fig. 6 shows that the initial inference steps are not used in the model merging process. This indicates that DFS relies heavily on prior knowledge to enhance the performance of the merged model. Furthermore, the experimental results demonstrate that DFS has less impact on performance improvement compared to PS and is not fully optimized. Therefore, research focusing on automating and optimizing DFS could make it a powerful tool for model merging.

## Qualitative Results
- ### Case Study of EvoLLM-JP-v1-7B
<p align="center">
  <img src="./figures/case study of evollm-jp.png" alt="." width="500" height="900" > 
</p>


- ### Case Study of EvoVLM-JP
<p align="center">
  <img src="./figures/case study of evovlm-jp.png" alt="." width="500" height="600" > 
</p>

## References

The Open LLM Leaderboard [https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)

Mergekit [https://github.com/arcee-ai/mergekit](https://github.com/arcee-ai/mergekit)

Ha, David., [A Visual Guide to Evolution Strategies](https://blog.otoro.net/2017/10/29/visual-evolution-strategies/). 2017.

Zhengqi Xu, Ke Yuan, Huiqiong Wang, Yong Wang, Mingli Song, Jie Song., [Training-Free Pretrained Model Merging](https://arxiv.org/abs/2403.01753). CVPR., 2024.

Sakana.ai., [Evolving New Foundation Models: Unleashing the Power of Automating Model Development](https://sakana.ai/evolutionary-model-merge/)., 2024.
