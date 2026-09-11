# Cross-Lingual Audio Deepfake Detection: Sensitive Layer Summarization on Frozen XLS-R for Low-Resource Hindi Speech

**Authors**: Anonymous Submission  
**Date**: September 2026  
**Target Venue**: Interspeech / ACM Multimedia / IEEE SLT  

---

## Abstract
Recent advances in neural voice cloning and text-to-speech synthesis have dramatically increased the threat posed by audio deepfakes, particularly in non-English linguistic regions where dedicated spoofing datasets and detection benchmarks are scarce. In this work, we evaluate the cross-lingual generalization and parameter-efficient adaptation of **Sensitive Layer Summarization (SLS)** on top of a frozen **Wav2Vec2 XLS-R 300M** backbone for Hindi audio deepfake detection. 

We formalize a rigorous benchmark on the SEA-Spoof dataset consisting of a balanced, held-out test split of 500 samples (250 genuine, 250 spoofed). Evaluating an SLS detector pretrained exclusively on English reveals an asymmetric cross-lingual gap: genuine Hindi speech is identified with 98.00% recall, but 59.6% of Hindi deepfakes evade detection under default classification thresholds, resulting in 69.20% accuracy and 10.60% Equal Error Rate (EER). 

To bridge this gap without risking catastrophic forgetting or requiring large-scale retraining, we investigate few-shot episodic adaptation where only the SLS head (~23M parameters) receives gradient updates while the 315M-parameter self-supervised backbone remains completely frozen. Fine-tuning on as few as $N=10$ clips per class (20 audio clips, <3s training time) cuts the EER by 81.4% to $1.97 \pm 0.67\%$ and boosts accuracy to $97.53 \pm 0.68\%$. With $N=50$ samples per class, the detector achieves near-perfect discrimination: **99.20% accuracy**, **0.40% EER**, and **100% precision on genuine speech** across three disjoint random seeds. Our findings prove that multilingual self-supervised representations encode rich language-agnostic acoustic artifacts that can be steered to non-English targets with minimal supervisory overhead.

---

## 1. Introduction
Synthetic speech generated via modern diffusion and neural vocoder architectures has made impersonation and fraudulent audio generation easily accessible. While detection systems trained on benchmark corpora like ASVspoof achieve sub-1% Equal Error Rates on in-domain English trials, real-world deployment faces severe distributional shifts—chief among them being linguistic diversity. 

Most audio deepfake research has focused on English. Non-English and Indic languages like Hindi (spoken by over 600 million people) remain critically underserved. Naive deployment of English-trained deepfake detectors in Hindi contexts risks high false acceptance rates, leaving communities vulnerable to voice cloning scams and synthetic misinformation.

To address this challenge without the prohibitive cost of curating massive language-specific deepfake corpora, this paper examines:
1. **The nature and magnitude of the cross-lingual transfer gap**: How does an English-trained sensitive layer summarization head behave when exposed to Hindi speech?
2. **Few-shot parameter-efficient adaptation**: Can minimal supervision ($N=10$ or $N=50$ clips) successfully recalibrate the decision surface on top of a frozen foundation model?
3. **Layer-wise summarization**: How does gating intermediate transformer states from a multilingual foundation model (XLS-R 300M) isolate spoofing cues across linguistic boundaries?

---

## 2. Architecture & Methodology

### 2.1 Frozen Multilingual Self-Supervised Backbone
We utilize **Wav2Vec2 XLS-R 300M** (`facebook/wav2vec2-xls-r-300m`), a self-supervised model pretrained on 436,000 hours of unlabelled speech across 128 languages, including Hindi. 
- Input raw audio at 16 kHz is fixed to length $L = 64,600$ samples (~4.04s).
- The temporal convolutional feature encoder generates latent feature representations.
- The 24-layer transformer encoder produces hidden states $H_l \in \mathbb{R}^{B \times T \times 1024}$ for each layer $l \in \{1, \dots, 24\}$, where $T = 201$.
- **Freezing constraint**: During all fine-tuning stages, the entire backbone parameter set $\Theta_{\text{backbone}}$ is frozen ($\nabla_{\Theta} = 0$).

### 2.2 Sensitive Layer Summarization (SLS) Head
Rather than extracting only the final transformer layer (which is heavily biased toward high-level semantic speech recognition), the SLS architecture exploits the observation that synthetic audio artifacts—such as phase incoherence, spectral blurring, and vocoder harmonic mismatch—manifest across different depths of the transformer hierarchy.

1. **Gated Layer Weighting**:
   Each layer's hidden state $H_l$ is projected through a trainable linear layer and sigmoid activation:
   $$\alpha_l = \sigma(W_0 \cdot \bar{H}_l + b_0) \in [0, 1]$$
   where $W_0 \in \mathbb{R}^{1024 \times 1}$ and $\bar{H}_l$ is the temporally pooled representation.

2. **Layer Fusion**:
   $$H_{\text{fused}} = \sum_{l=1}^{24} \alpha_l \cdot H_l \in \mathbb{R}^{B \times T \times 1024}$$

3. **2D Convolutional and Downsampling Block**:
   $H_{\text{fused}}$ is treated as a single-channel 2D feature map $(B, 1, T, 1024)$:
   $$Z = \text{MaxPool2D}(\text{SELU}(\text{BatchNorm2D}(H_{\text{fused}})))$$
   with kernel size $(3, 3)$ and stride $(3, 3)$, reducing the temporal-feature matrix to a flattened dimension of 22,847.

4. **Classifier MLP**:
   $$y = \text{LogSoftmax}(W_3 \cdot \text{SELU}(W_1 \cdot Z + b_1) + b_3)$$
   where $W_1 \in \mathbb{R}^{22847 \times 1024}$ and $W_3 \in \mathbb{R}^{1024 \times 2}$.

---

## 3. Experimental Protocol

### 3.1 Dataset: SEA-Spoof (Hindi Subset)
We sample from the `Jack-ppkdczgx/SEA-Spoof` benchmark, targeting genuine and synthetic utterances marked `language == 'hi'`.
- 400 bonafide clips and 400 spoof clips are extracted into a localized parquet split (800 total clips).
- Spoofing mechanisms encompass modern neural text-to-speech (TTS) and voice conversion (VC) algorithms.

### 3.2 Benchmark Protocol & Disjointness
- **Held-Out Test Set**: 250 bonafide and 250 spoof clips ($N_{\text{test}} = 500$), sampled with seed 42.
- **Few-Shot Candidate Pool**: The remaining 300 clips (150 bonafide, 150 spoof).
- **Strict Disjointness**: The test set is guaranteed to have 0 overlap with any training set.
- **Nested Few-Shot Seeds**:
  - Experiments are evaluated at $N=10$ (20 training samples) and $N=50$ (100 training samples) across three independent random seeds (1000, 1001, 1002).
  - For each seed, the $N=10$ pool is a strict mathematical subset of the $N=50$ pool.

### 3.3 Training Details
- Optimizer: Adam ($lr = 10^{-5}$, weight decay $10^{-4}$).
- Criterion: Negative Log-Likelihood Loss with class weighting $w = [0.1, 0.9]$ (spoof=0, bonafide=1).
- Epochs: 10 epochs per seed.
- Memory Optimization: On-the-fly batched tensor loading ensuring memory footprint remains under 100 MB.

---

## 4. Results & Analysis

### 4.1 Quantitative Comparison
Table 1 presents the performance of Zero-Shot transfer versus Few-Shot adaptation evaluated on the 500-sample Hindi test set.

**Table 1: Evaluation Metrics on the Held-Out Hindi Test Set ($N=500$)**

| Model Setting | Test Accuracy (%) | EER (%) | Precision (%) | Recall (%) | F1 Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Zero-Shot (English Checkpoint)** | **69.20%** | **10.60%** | **62.18%** | **98.00%** | **76.09%** |
| **Few-Shot $N=10$ (3 Seeds)** | **97.53 $\pm$ 0.68%** | **1.97 $\pm$ 0.67%** | **96.39 $\pm$ 1.55%** | **98.80 $\pm$ 0.86%** | **97.57 $\pm$ 0.65%** |
| **Few-Shot $N=50$ (3 Seeds)** | **99.20 $\pm$ 0.00%** | **0.40 $\pm$ 0.01%** | **100.00 $\pm$ 0.00%** | **98.40 $\pm$ 0.00%** | **99.19 $\pm$ 0.00%** |

*(Metrics computed with Bonafide as positive class).*

---

### 4.2 Analysis of the Zero-Shot Domain Shift

**Zero-Shot Confusion Matrix ($N=500$)**:
```
                     Predicted Spoof (0)    Predicted Bonafide (1)
  True Spoof (0)                     101                       149
  True Bonafide (1)                    5                       245
```

The zero-shot confusion matrix illustrates a notable phenomenon:
- **Low False Rejection Rate (FRR)**: Only 5 out of 250 bonafide Hindi utterances are misclassified as spoof (2.0% error).
- **High False Acceptance Rate (FAR)**: 149 out of 250 spoofed Hindi utterances are accepted as bonafide (59.6% error).

When inspecting the score distributions:
- Ground truth Bonafide log-probability: $\text{mean} = -0.059$ ($P \approx 0.94$).
- Ground truth Spoof log-probability: $\text{mean} = -0.673$ ($P \approx 0.51$).

Because the model was trained on English data with an asymmetric class penalty, the log-odds output for Hindi speech is shifted positively. An argmax decision threshold of 0.5 is misaligned, creating a false-acceptance vulnerability. However, the receiver operating characteristic reveals that the classes remain well-separated in the latent space, achieving an **EER of 10.60%** at threshold $-0.215$.

---

### 4.3 Few-Shot Adaptation Dynamics

**$N=50$ Few-Shot Confusion Matrix ($N=500$, Seed 0)**:
```
                     Predicted Spoof (0)    Predicted Bonafide (1)
  True Spoof (0)                     250                         0
  True Bonafide (1)                    4                       246
```

Adapting the SLS head with minimal supervision yields rapid, stable convergence:
1. **$N=10$ per class (20 clips total)**:
   - Average loss drops from $0.48$ at epoch 1 to $0.01$ by epoch 10.
   - The EER plunges from $10.60\%$ down to $1.97\%$.
   - Spoof detection accuracy increases from $40.4\%$ to $96.8\%$.
2. **$N=50$ per class (100 clips total)**:
   - False alarms on spoofed audio are **completely eliminated** ($250 / 250$ spoof samples detected across all three seeds).
   - Only 4 bonafide samples out of 250 are falsely rejected.
   - Variance across random seeds reaches **$\pm 0.00\%$** in accuracy and $\pm 0.01\%$ in EER.

---

## 5. Conclusion & Future Directions
This paper investigated the cross-lingual transferability of Sensitive Layer Summarization on frozen XLS-R representations for Hindi audio deepfake detection. While zero-shot deployment from English checkpoints exhibits significant threshold displacement, fine-tuning only the lightweight SLS head with as few as 10–50 samples per class provides near-perfect detection (99.20% accuracy, 0.40% EER). 

Our results confirm that multilingual self-supervised foundation models preserve generalizable acoustic artifacts that require only lightweight linear steering to protect low-resource and non-English communities from synthetic voice deception. Future work will extend this protocol across other Indic languages (Tamil, Bengali, Telugu) and evaluate robustness against lossy telephony codecs.

---

## References
1. Sukhdeve, Y., et al. "XLS-R-SLS: Selective Layer Summarization for Audio Deepfake Detection," HuggingFace Model Hub, 2024.
2. Babu, A., et al. "XLS-R: Self-supervised Cross-lingual Speech Representation Learning at Scale," Interspeech 2022.
3. Zhang, Y., et al. "Sensitive Layer Summarization for Synthetic Speech Detection," ACM Multimedia, 2024.
4. Yi, C., et al. "SEA-Spoof: A Multilingual Synthetic Speech Dataset for Southeast and South Asian Languages," 2024.
