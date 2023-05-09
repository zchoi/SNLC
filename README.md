# SNLC 🍃 
> [**Learning Visual Question Answering on Controlled Semantic Noisy Labels**](https://www.sciencedirect.com/science/article/pii/S0031320323000407) [**Pattern Recognition 2023**] <br>
> Haonan Zhang, Pengpeng Zeng, Yuxuan Hu, Jin Qian, Jingkuan Song and Lianli Gao
## Overview 
Visual Question Answering (VQA) has made great progress recently due to the increasing ability to understand and encode multi-modal inputs based on deep learning. However, existing VQA models are usually based on assumptions of clean labels, and it is contradictory to real scenarios where labels are expensive and inevitably contain noises. In this paper, we take the lead in addressing this issue by establishing the first benchmark of controlled semantic noisy labels for VQA task, evaluating existing methods, and coming up with corresponding solutions. 

<p align="center">
    <img src=pic\introduction.png  width="45%">
    <img src=pic\distribution.png  width="50%" height="20%">
    <span><b>Figure 1. Left: Motivation. Right: Visualization of semantic space of candidate answers by t-SNE.</b></span>
</p>


## VQA-Noise v1 and VQA-Noise v2 Benchmarks

We adopt semantic similarity to guide the generation of noisy labels in a controlled manner and build new datasets: VQA-Noise v1 and VQA-Noise v2. Specifically, we first adopt the pre-trained word embedding (*e.g.*, [BERT](https://huggingface.co/bert-base-uncased) and [Glove](https://nlp.stanford.edu/projects/glove/)) to embed all candidate answers into an answer-based semantic embedding space (**Fig.1 Right**). Then, we randomly sample image-question pairs at a sample rate of $\boldsymbol{p}\in[0.2, 0.4, 0.6, 0.8]$. The ground-truth answer of each sampled pair is replaced with a pseudo label, which is randomly selected from the top-K answers that are semantically similar to the ground-truth answer

> **Note**: The generated noisy annotations of [**VQA-Noise v1**](https://drive.google.com/drive/folders/1sdNvgn9zoBjDOhJA9lFsVba3ZM6ac7xT?usp=sharing) and [**VQA-Noise v2**](https://drive.google.com/drive/folders/1sdNvgn9zoBjDOhJA9lFsVba3ZM6ac7xT?usp=sharing) are available.

<p align="center">
    <img src=pic\noise_examples.png  width="100%">
    <span><b>Figure 2. Selected examples of generated semantic noisy labels from VQA-Noise v2. </b></span>
</p>

------

Below we demonstrate the performance degradation of VQA models (*i.e.*, [UpDn](https://github.com/hengyuan-hu/bottom-up-attention-vqa)) on our noisy VQA benchmarks, which shows the vulnerabilities of existing approaches to noisy learning.

<p align="center">
    <img src=pic\performance.png  width="100%">
    <span><b>Figure 3. Performance degradation of UpDn model on VQA-Noise v2. </b></span>
</p>

## Method
Overview of general VQA models with our proposed SNLC for learning with controlled semantic noisy labels. Specifically, SNLC includes **Semantic Cross-Entropy (SCE)** loss and **Semantic Embedding Contrastive (SEC)** loss. SCE loss tolerates a wide range of noisy labels and SEC loss obtains a robust representation in answer embedding space.
<p align="center">
    <img src=pic\framework.png alt="Framework"  width="100%">
    <span><b>Figure 4. Overall Framework.</b></span>
</p>

------
> **Note**: It should be straightforward to adopt ```SCE``` and ```SEC``` in pipeline when training your noisy-robust model on VQA-Noise v1/v2.

```python
from SCE import SCE
from SEC import SEC

sce_ = SCE()
sec_ = SEC(margin=0.2)

for batch in dataset.train:
  v, q, a = (b.to(device) for b in batch)    # feature represnetations of image, question, and label

  # forward-backward step
  pred, sim = model(v, q, a)
  loss_1 = instance_bce_with_logits(pred, a) # traditional bce loss
  loss_2 = sce_(pred, a, topk=4)             # Semantic cross-entropy (SCE) loss.
  loss_3 = sec_(sim, None, a)                # Semantic embedding contrastive (SEC) loss.
  loss = loss_1 + α*loss_2 + β*loss_3
  loss.backward()
  optim.step()
  optim.zero_grad()

```
## Experiments
Performance for noisy learning on BAN model and the effect of the proposed SNLC. All values are reported as accuracy (%). For more experimental results, please refer to our [paper](https://www.sciencedirect.com/science/article/pii/S0031320323000407)

<p align="center">
    <img src=pic\ban.png alt="Framework"  width="100%">
</p>


## Citation
```bibtex
@article{ZHANG2023109339,
title = {Learning visual question answering on controlled semantic noisy labels},
journal = {Pattern Recognition},
volume = {138},
pages = {109339},
year = {2023},
issn = {0031-3203},
doi = {https://doi.org/10.1016/j.patcog.2023.109339},
url = {https://www.sciencedirect.com/science/article/pii/S0031320323000407},
author = {Haonan Zhang and Pengpeng Zeng and Yuxuan Hu and Jin Qian and Jingkuan Song and Lianli Gao},
keywords = {Visual question answering, Noisy datasets, Semantic labels, Contrastive learning},
}
```
