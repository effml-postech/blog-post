---
type: docs
bookToc: True
weight: 1
---

# Beyond Language Models: Byte Models are Digital World Simulators

*Authors: Shangda Wu, Xu Tan, Zili Wang, Rui Wang, Xiaobing Li, Maosong Sun* <br>
*Links: [Paper](https://arxiv.org/abs/2402.19155), [GitHub](https://github.com/sanderwood/bgpt), [Hugging Face](https://huggingface.co/sander-wood/bgpt), [Official Project Page](https://byte-gpt.github.io/)*

*Posted by Dohun Kim and Yeongwoo Kim*

## Introduction

Byte models take traditional language models to the byte level, treating all digital data and operations as fundamentally byte-based. These models process data from different modalities (such as text, images, and audio) uniformly as bytes, making them versatile in a wide digital environment. However, recent research on byte models has been limited, primarily focusing on narrow tasks and overlooking their broader potential in simulating the digital world.

![bgpt_framework](framework.JPG) <br>
*[Figure 1](https://byte-gpt.github.io/): The bGPT framework simulates digital systems using native binary data. It integrates diverse data types into a single model by treating everything as a byte sequence.*

To address this issues, the authors propose bGPT, which operates at the byte level and efficiently processes byte sequences. Through comprehensive evaluations across various modalities, bGPT demonstrates performance comparable to specialized models. Moreover, bGPT opens up new possibilities for simulating algorithms and hardware operations. By learning to predict the next byte, bGPT provides a deeper understanding of the intricate patterns in the digital world.

The main contributions of this paper are as follows:

- **bGPT**, a model with *next-byte prediction* is presented to simulate digital systems
- **Hierarchical Transformer architecture** is adapted to handle byte sequences efficiently.
- **In-depth analysis** of bGPT's performance on text, audio, and image data is provided.
- **Novel benchmarks** are introduced to show bGPT's capabilities for digital world simulation.

## bGPT Framework

### Model Architecture

Learning patterns in digital systems at the byte level offers a unified approach for integrating various data types. However, the high granularity of bytes leads to long sequences, which significantly increase computational costs due to the quadratic scaling of self-attention. This limits the efficiency and scalability of processing binary data.

<p align="center">
  <img src=architecture.JPG width="500">
</p>

*[Figure 2](https://byte-gpt.github.io/): The hierachical Transformer architecture of bGPT. It segments byte sequences into patches, to balance the need for long sequences and computational efficiency.*

To address this issue, the authors adapted a hierarchical structure for bGPT, enabling efficient handling of long byte sequences. This structure segments a sequence of byte {{< katex >}}B{{< /katex >}} of length {{< katex >}}T{{< /katex >}} into a sequence of patches {{< katex >}}\mathcal{P}{{< /katex >}}, where each patch contains exactly {{< katex >}}S{{< /katex >}} bytes, i.e., {{< katex >}}\mathcal{P}{{< /katex >}} includes {{< katex >}}N = \left\lceil \frac{T}{S} \right\rceil{{< /katex >}} patches. If {{< katex >}}T \mod S \neq 0{{< /katex >}}, the last patch is padded with `<eop>` (end-of-patch) token.

#### Components of bGPT

- **Byte encoding**: Each byte is one-hot encoded into a 257-dimensional vector, including all possible byte values and `<eop>` token. Thus, each patch is viewed as a matrix of size {{< katex >}}S \times 257{{< /katex >}}.
- **Linear Projection Layer** maps the flattened patch into a dense vector of hidden size {{< katex >}}H{{< /katex >}}, enabling more efficient processing of byte sequence.
- **Patch-Level Decoder** autoregressively predicts the features of the next patch, thereby learning the structural patterns of the entire dataset.
- **Byte-Level Decoder** takes the predicted patch features and autoregressively reconstruct the bytes within each patch. This process is repeated for all patches to generate the output byte sequence.

### Training Objectives

#### Generative Modeling

This approach requires the model to predict the next byte in a given byte sequence. The model takes the byte sequence {{< katex >}}B = \{b_1, b_2, \ldots, b_T\}{{< /katex >}} as input and utilizes all previous byte information to predict the next byte {{< katex >}}b_{i+1}{{< /katex >}} at each position.

As a loss function, the negative log likelihood of the next byte at each step is minimized. This encourages the model to maximize the likelihood of the actual occurrence of the next byte.
{{< katex display=true >}}
  \mathcal{L}_{\text{GEN}}(\theta) = - \sum_{i=1}^{T-1} \log p(b_{i+1} \mid b_1, b_2, \ldots, b_i; \theta)
{{< /katex >}}

#### Classification

Based on the knowledge acquired through generative modeling, bGPT can also be applied to classification tasks for labeled datasets. In this process, the model takes a byte sequence as input and predicts the category to which that sequence belongs.
For classification tasks, the loss function used is the cross-entropy loss, which ensures that the model accurately outputs the prediction probabilities for each category.

{{< katex display=true >}}
  \mathcal{L}_{\text{CLF}}(\theta) = - \sum_{k=1}^{K} y_k \log p(y_k \mid B; \theta)
{{< /katex >}}

These training objectives enable bGPT to understand various byte-based data and accurately mimic digital patterns of the real world. The combination of generative approaches and classification capabilities grants the model the flexibility to tackle a diverse range of problems. Through this, the model can go beyond simple pattern recognition to play a crucial role in predicting and analyzing the operations of complex digital systems.

***

## Applications

### Digital Media Processing

bGPT is used for processing various types of digital media data such as text, audio, and images. This model performs learning targeted at media files through generative modeling, transforming the data into features and subsequently performing classification tasks based on these features.

For example, audio files are converted to and processed in WAV format, while images are processed at a low resolution in BMP format. By utilizing these standardized datasets, bGPT can develop a generalized understanding of various media types.

### Algorithm and Hardware Simulation

bGPT is particularly useful for tasks such as data conversion and CPU state modeling. This means bGPT can learn the digital conversion process and simulate the operation of CPUs to predict the state of the CPU after various commands are executed.

For example, in the task of converting the music data format from ABC notation to MIDI format, bGPT learns to transform text-based music scores in ABC notation into binary performance signals in MIDI. Additionally, this model is also capable of performing the reverse conversion from MIDI back to ABC notation.

## Experiments

### Digital Media Processing (well-studied tasks)

**Experiment Overview**
To assess the flexibility and versatility of the bGPT model, experiments with various types of digital media data were conducted. This involved handling a wide range of file types including text, audio, and image data, with the aim to measure the model's ability to process these types and to see how well bGPT generalizes compared to specialized models. The experiment included both generative modeling and classification tasks.

<!-- **Experimental Data**
The datasets used in the experiment included:

- **Text**: Wikipedia data and AG news dataset.
- **Audio**: LibriSpeech and Speech Commands v2 dataset.
- **Images**: ImageNet and CIFAR-10 dataset.

These datasets are ideal resources for evaluating the diverse media processing capabilities of bGPT.

**Experimental Setup** 
The bGPT model was trained under various settings:

- **bGPTimage**: Trained exclusively with image data (ImageNet).
- **bGPTlibri**: Trained exclusively with text data (Wikipedia).
- **bGPTmix**: Trained with a mix of all the above datasets.

Each model was fine-tuned for specific types of classification and generative tasks post-pretraining, providing a direct comparison of each model's performance. -->

#### Experimental Setup

| Model      | Modality | Pre-training dataset |
| ---------- | -------- | -------------------- |
| bGPTrandom | (None)   | (None, randomly initialized) |
| bGPTwiki   | Text     | Wikipedia            |
| bGPTimage  | Image    | ImageNet             |
| bGPTlibri  | Speech   | LibriSpeech          |
| bGPTsignal | Image+Speech | ImageNet+LibriSpeech |
| bGPTmix    | Text+Image+Speech | Wikipedia+ImageNet+LibriSpeech |

**Result and Analysis**

- **Text Processing**: bGPTwiki showed high classification accuracy on the AG News dataset, indicating bGPT's strong performance in text-based tasks.
- **Audio Processing**: bGPTlibri demonstrated excellent performance on the Speech Commands v2 dataset, showcasing its high potential in audio processing.
- **Image Processing**: bGPTimage recorded high accuracy on the CIFAR-10 dataset but showed somewhat lower performance on ImageNet. This suggests that while bGPT works well with relatively simple images, it may have limitations with more complex images.
  
  | Model           | \| AG News (4 classes) \| |           | CIFAR-10 (10 classes) \| |           | Speech Commands v2 (36 classes) \| |
|-----------------|----------------------------|-----------|----------------------------|-----------|------------------------------------|
|                 | BPB     | Acc (%)           | BPB       | Acc (%)                    | BPB       | Acc (%)                            |
|-----------------|---------|-------------------|-----------|----------------------------|-----------|------------------------------------|
| bGPT_random     | 1.3496  | 84.74             | 3.4928    | 76.73                      | 1.5414    | 92.43                              |
| bGPT_wiki       | **1.0639**  | **92.49**             | 3.6663    | 77.02                      | 1.5719    | 93.56                              |
| bGPT_image      | 1.4179  | 83.16             | **3.1234**    | **88.69**                      | 1.5326    | 93.91                              |
| bGPT_libri      | 1.3993  | 83.59             | 3.3345    | 83.51                      | **1.4818**    | **96.03**                              |
| bGPT_signal     | 1.4058  | 83.80             | 3.1554    | 87.65                      | 1.4898    | 95.66                              |
| bGPT_mix        | 1.0935  | 91.75             | 3.2279    | 84.32                      | 1.5086    | 95.09                              |
| Baselines       | 0.9237  | 94.50             | —         | **98.13**                      | —         | **98.11**                              |


### Algorithm and Hardware Simulation (underexplored tasks)

**Experiment Overview** 
One of the unique capabilities of the bGPT model is its ability to simulate the operations of algorithms and hardware. This experimental section assesses how bGPT handles complex data conversion processes and CPU state modeling tasks. These capabilities are particularly significant in the fields of cybersecurity, system diagnostics, and hardware optimization.

**Experiment Methods** 
bGPT's performance was evaluated in the following two key areas:

- **Data Conversion**: This experiment evaluates whether bGPT can learn the process of converting ABC music notation into MIDI format. The task tests how bGPT models complex algorithms and their ability to convert actual music files.
- **CPU State Modeling**: CPU state modeling assesses how bGPT predicts and updates the state of a CPU based on a given set of machine instructions. This is particularly useful for understanding and predicting hardware operations.

**Results and Analysis** 


<p float="center">
  <img src="dataconversion.JPG" alt="Image 1" width="400" />
  <img src="cpumodeling.JPG" alt="Image 2" width="400" /> 
</p>

- **Data Conversion Performance**: bGPT performed the conversion between MIDI and ABC notation with high accuracy. Notably, it also showed high accuracy in converting MIDI back to ABC notation, indicating that bGPT successfully learned the inherent structures and patterns of the data.
- **CPU State Modeling Performance**: bGPT accurately predicted the resulting state of CPUs from an initial state across a variety of CPU instructions. It achieved over 99% accuracy even with complex instruction sequences, demonstrating bGPT's detailed understanding of the internal workings of hardware.

***

## Conclusion and Future Work

bGPT has proven to be a powerful model capable of effectively processing various types of digital media data. Particularly, this model can be flexibly applied to different types of data and has demonstrated performance that can compete with models pretrained on specific datasets. These results show that bGPT can be extremely useful in solving a wide range of real-world problems.

Moreover, bGPT has proven its ability to go beyond simply processing data, successfully modeling and simulating the operations of complex algorithms and hardware. This capability will be particularly valuable in fields related to technical problem solving and new hardware design.

bGPT extends deep learning to binary data processing through byte prediction. Experiments have demonstrated bGPT's strong scalability in native binary data modeling.

Future research directions for byte models include:

- Reducing training costs to make byte model training more feasible.
- Expanding the model and dataset sizes to accommodate a wider range of native binary data and handle larger digital media files such as high-resolution images and videos.
- Improving performance in underexplored tasks involving native binary data across various application domains.

## References

[Beyond Language Models: Byte Models are Digital World Simulators (arXiv)](https://arxiv.org/abs/2402.19155) <br>
