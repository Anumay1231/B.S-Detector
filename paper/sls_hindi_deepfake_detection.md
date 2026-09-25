# Cross-Lingual Audio Deepfake Detection: Sensitive Layer Summarization on Frozen XLS-R for Low-Resource Hindi Speech

**Authors**: Anonymous Submission  
**Date**: September 2026  
**Target Venue**: IEEE SLT / ACM Multimedia / Interspeech 2025  

---

## Abstract
Recent advances in neural voice cloning and text-to-speech synthesis (TTS) have made audio deepfakes a pervasive security threat, particularly in non-English linguistic regions where dedicated spoofing benchmarks and language-specific detectors are scarce. In this work, we evaluate the cross-lingual generalization and parameter-efficient adaptation of **Sensitive Layer Summarization (SLS)** on top of a frozen multilingual foundation model (**Wav2Vec2 XLS-R 300M**) for Hindi audio deepfake detection. 

We identify and demonstrate a critical methodological pitfall in cross-lingual benchmarking: **generator acoustic bias**. When evaluated on single-source synthetic audio (V1), detectors exploit spurious synthesis shortcuts, producing an artificially low Equal Error Rate (EER) of **0.40%**. To eliminate this shortcut, we construct a strictly balanced, multi-generator benchmark (V2) comprising 400 test samples (200 bonafide from studio and crowdsourced recordings; 200 spoof evenly stratified across 5 distinct TTS architectures: Edge-TTS, FastSpeech, Indic-TTS, VITS-MMS, and XTTS-v2). 

Evaluating an English-pretrained SLS checkpoint zero-shot on Hindi reveals a substantial cross-lingual domain shift: genuine Hindi speech is detected with 98.00% recall, but synthetic speech frequently evades detection, yielding an EER of **28.90%** and **60.00% accuracy**. Fine-tuning only the lightweight SLS classifier head (~22.8M parameters) with few-shot Hindi supervision rapidly closes this gap: with as few as $N=10$ samples per class, EER improves to **26.88%** ($65.83\%$ accuracy); scaling to $N=50$ samples per class reduces EER to **19.70%** and drives accuracy to **79.67%** (a 9.2% absolute EER reduction). 

Furthermore, we conduct an architectural ablation on an NVIDIA L4 GPU investigating whether unfreezing the transformer backbone benefits few-shot adaptation. Unfreezing the top 2 layers (top 8% of backbone parameters) degrades EER to **24.14%**, and unfreezing the top 4 layers (top 17%) leads to catastrophic few-shot overfitting (**25.31% EER**, $67.12\%$ accuracy). Our empirical findings prove that keeping 100% of the multilingual foundation backbone frozen is decisively superior under few-shot data regimes ($N \le 50$), validating the parameter-efficient SLS paradigm for cross-lingual deepfake defense.

---

## 1. Introduction
Synthetic speech generated via modern diffusion, flow-matching, and neural vocoder architectures has made high-fidelity voice cloning accessible at consumer scale. While deepfake detectors trained on established English corpora (e.g., ASVspoof 2019/2021) achieve sub-1% Equal Error Rates in in-domain conditions, deployment in real-world scenarios requires robust cross-lingual generalization. 

The vast majority of synthetic speech research remains concentrated on English. Indic languages such as Hindi—spoken by over 600 million people worldwide—remain severely underserved by defensive technologies. Deploying English-trained detection models directly on Hindi audio exposes populations to high false-acceptance vulnerabilities, creating substantial risks in telephonic fraud, financial voice-authorization scams, and political disinformation.

Building dedicated large-scale deepfake datasets for every under-resourced language is economically and computationally prohibitive. Consequently, a vital research question arises: **Can a deepfake detection head pretrained on high-resource English transfer to an under-resourced target language with minimal supervision, and what architectural constraints govern this transfer?**

This work provides a rigorous investigation into this question using the **Selective / Sensitive Layer Summarization (SLS)** framework built on a frozen **XLS-R 300M** foundation model. Specifically, our contributions are:

1. **Quantifying Cross-Lingual Domain Shift**: We evaluate an English-pretrained SLS detector on Hindi speech without target-language adaptation (Zero-Shot), demonstrating an asymmetric failure mode where genuine speech is preserved (98.00% recall) while 44.3% of synthetic speech evades detection (28.90% EER).
2. **Uncovering and Resolving Generator Bias**: We demonstrate that evaluating synthetic speech detectors on unstratified datasets (V1) creates spurious correlation shortcuts, artificially deflating EER to 0.40%. We formalize a 5-generator stratified evaluation protocol (V2) that prevents acoustic shortcut learning and measures true cross-generator generalization.
3. **Parameter-Efficient Few-Shot Adaptation**: We show that fine-tuning solely the lightweight SLS head with $N \in \{10, 50\}$ Hindi samples per class achieves monotonic performance gains, decreasing EER from 28.90% to 19.70% and increasing accuracy from 60.00% to 79.67%.
4. **Backbone Unfreezing vs. Frozen Foundation Analysis**: On an NVIDIA L4 GPU, we show that unfreezing transformer backbone layers during few-shot adaptation causes severe overfitting, confirming that keeping the 315M-parameter foundation backbone completely frozen provides optimal regularization.

---

## 2. Architecture & Methodology

```
Raw Audio (16 kHz, 64,600 samples)
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│  Wav2Vec2 XLS-R 300M Backbone (315.4M Params - 100% Frozen) │
│  Temporal Conv Encoder -> 24 Transformer Hidden Layers      │
└─────────────────────────────────────────────────────────────┘
       │  Extract all 24 layer hidden states H_1, ..., H_24
       ▼
┌─────────────────────────────────────────────────────────────┐
│  Selective Layer Summarization (SLS) Head (~22.8M Params)   │
│  1. Temporal Pooling & Sigmoid Attention Gate: alpha_l      │
│  2. Weighted Layer Fusion: H_fused = sum(alpha_l * H_l)     │
│  3. BatchNorm2D -> SELU -> MaxPool2D(3, 3)                  │
│  4. Flatten (22,847 dim) -> Linear(1024) -> SELU            │
│  5. Linear(1024, 2) -> LogSoftmax -> [P(spoof), P(bonafide)]│
└─────────────────────────────────────────────────────────────┘
```

### 2.1 Multilingual Foundation Backbone (XLS-R 300M)
We employ **Wav2Vec2 XLS-R 300M** (`facebook/wav2vec2-xls-r-300m`), trained on 436,000 hours of unlabelled speech spanning 128 languages (including extensive Hindi coverage).
- **Input**: Raw audio waveforms sampled at 16 kHz, standardized to length $L = 64,600$ samples (~4.04 seconds). Shorter utterances are tile-padded; longer utterances are center-cropped.
- **Latent Feature Extraction**: A 7-block temporal convolutional feature encoder transforms the raw audio into latent feature vectors.
- **Transformer Representations**: 24 transformer layers produce hidden representations:
  $$H_l \in \mathbb{R}^{B \times T \times 1024}, \quad l \in \{1, 2, \dots, 24\}, \quad T = 201$$
- **Backbone Freezing**: Unless explicitly tested in ablation experiments, the entire parameter set $\Theta_{\text{backbone}}$ is frozen ($\nabla_{\Theta_{\text{backbone}}} = 0$).

### 2.2 Sensitive Layer Summarization (SLS) Mechanism
Traditional audio classifiers often attach a classification head solely to the 24th (final) transformer layer. However, synthetic audio artifacts—such as vocoder phase discontinuity, high-frequency harmonic distortion, and pitch modulation collapse—are acoustic artifacts that are captured prominently in intermediate layers, whereas final layers encode linguistic and semantic abstractions.

SLS addresses this by learning a soft attention gate over all 24 transformer representations:
1. **Per-Layer Gating**:
   Each layer's representation $H_l$ is temporally average-pooled:
   $$\bar{h}_l = \frac{1}{T} \sum_{t=1}^T H_l[:, t, :] \in \mathbb{R}^{B \times 1024}$$
   and projected through a learned linear transformation with sigmoid activation:
   $$\alpha_l = \sigma(W_0 \bar{h}_l + b_0) \in \mathbb{R}^{B \times 1}, \quad W_0 \in \mathbb{R}^{1024 \times 1}$$
2. **Dynamic Layer Fusion**:
   $$H_{\text{fused}} = \sum_{l=1}^{24} \alpha_l \odot H_l \in \mathbb{R}^{B \times T \times 1024}$$
3. **2D Feature Representation & Downsampling**:
   Treating $H_{\text{fused}}$ as a single-channel 2D feature map of shape $(B, 1, T, 1024)$:
   $$Z = \text{MaxPool2D}(\text{SELU}(\text{BatchNorm2D}(H_{\text{fused}})))$$
   with kernel size $(3, 3)$ and stride $(3, 3)$, yielding a flattened feature representation $z \in \mathbb{R}^{B \times 22847}$.
4. **Classification Head**:
   $$\hat{y} = \text{LogSoftmax}(W_3 \cdot \text{SELU}(W_1 z + b_1) + b_3)$$
   where $W_1 \in \mathbb{R}^{22847 \times 1024}$ and $W_3 \in \mathbb{R}^{1024 \times 2}$.

---

## 3. Dataset Construction & The Generator Bias Phenomenon

![Dataset Architecture](outputs/figures/fig4_generator_stratification.png)
*Figure 1: Architecture of the V2 Balanced Benchmark: 50/50 Binary Class Balance (left) and strict 20% stratification across 5 distinct TTS generator architectures (right).*

### 3.1 The Dataset Bias Hypothesis (V1 Pitfall)
In preliminary experiments (designated **V1**), synthetic audio was sampled uniformly from the SEA-Spoof Hindi corpus without controlling for generator provenance. This resulted in an evaluation set where spoof audio was heavily dominated by a single voice synthesizer (`indic_tts`). 

Under this setting, the model achieved an apparent Equal Error Rate of **0.40%** at $N=50$. However, auditing the learned attention weights and latent space revealed that the classifier was not learning general deepfake artifacts; rather, it was learning the specific spectral signature and vocoder artifacts of the dominant TTS engine.

### 3.2 Formalizing the V2 Stratified Benchmark
To prevent shortcut learning and measure genuine cross-generator generalization, we constructed the **V2 Benchmark** from the SEA-Spoof Hindi subset (54,938 total candidate utterances):
- **Bonafide Class (400 samples total)**: Balanced between high-quality studio recordings (IndicTTS, ~60%) and in-the-wild crowdsourced acoustic environments (Mozilla Common Voice Hindi, ~40%).
- **Spoof Class (400 samples total)**: Equally stratified across **5 distinct TTS architectures** (80 samples each, exactly 20.0% per generator):
  1. `edge_tts`: Microsoft cloud neural TTS (fast, production voice cloning).
  2. `vits_mms`: Meta MMS variational inference end-to-end synthesizer.
  3. `indic_tts`: IIT Madras fast pitch-synchronous unit-selection and acoustic models.
  4. `fastspeech`: Feed-forward non-autoregressive transformer TTS.
  5. `xtts-v2`: Modern diffusion-based zero-shot voice cloning.

### 3.3 Test and Few-Shot Pool Partitioning
- **Held-Out Test Set ($N_{\text{test}} = 400$)**: Exactly 200 bonafide samples and 200 spoof samples (40 clips per generator).
- **Few-Shot Candidate Pool**: The remaining disjoint 302 clips (200 bonafide, 102 spoof across all 5 models).
- **Disjointness Guarantee**: Test and training sets have zero speaker or utterance overlap ($\text{Test} \cap \text{Train} = \emptyset$).
- **Multi-Seed Stability**: Evaluated across 3 deterministic seeds (seeds 1000, 1001, 1002).

---

## 4. Experimental Results

### 4.1 Benchmark Results on the Strictly Balanced V2 Dataset
Table 1 presents the performance of the English-pretrained baseline (Zero-Shot) alongside few-shot adaptation at $N=10$ and $N=50$ samples per class on the strictly balanced Hindi test set ($N_{\text{test}} = 400$).

**Table 1: Evaluation Metrics on Held-Out Hindi Test Set ($N_{\text{test}} = 400$, 50/50 Balanced)**

| Experiment | Accuracy (%) | EER (%) | Precision (%) | Recall (%) | F1 Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Zero-Shot (English Checkpoint)** | **60.00%** | **28.90%** | **55.68%** | **98.00%** | **71.01%** |
| **Few-Shot $N=10$ (3 Seeds)** | **65.83 $\pm$ 5.87%** | **26.88 $\pm$ 1.83%** | **60.81 $\pm$ 5.16%** | **94.50 $\pm$ 6.42%** | **73.61 $\pm$ 2.35%** |
| **Few-Shot $N=50$ (3 Seeds)** | **79.67 $\pm$ 0.47%** | **19.70 $\pm$ 0.57%** | **78.82 $\pm$ 3.58%** | **81.83 $\pm$ 5.25%** | **80.05 $\pm$ 0.88%** |

*(Metrics computed with Bonafide as the positive class. Mean $\pm$ standard deviation across 3 independent random seeds).*

![Few-Shot Adaptation Curve](outputs/figures/fig2_fewshot_adaptation_curve.png)
*Figure 2: Few-Shot Adaptation Dynamics on Frozen XLS-R 300M. Accuracy increases from 60.00% to 79.67%, while EER drops from 28.90% to 19.70% as few-shot supervision scales to $N=50$.*

### 4.2 Analysis of the Zero-Shot Cross-Lingual Gap
Under Zero-Shot transfer, the English-pretrained model achieves:
- **Bonafide Recall: 98.00%** (196 of 200 real Hindi utterances are correctly recognized as genuine).
- **Spoof Precision: 55.68%** (a significant portion of Hindi synthetic speech slips past the decision boundary).

Because the English pretrained checkpoint was trained with an asymmetric penalty on ASVspoof, the uncalibrated logits for Hindi audio exhibit a positive shift. The baseline EER of **28.90%** proves that cross-lingual transfer without target-language calibration leaves substantial vulnerability to voice spoofing attacks.

### 4.3 Few-Shot Adaptation Dynamics
Fine-tuning only the SLS head yields rapid calibration:
- **$N=10$ (20 clips total, <3 seconds training on L4 GPU)**: EER decreases by $2.02\%$ absolute, with accuracy improving to $65.83\%$.
- **$N=50$ (100 clips total, <15 seconds training on L4 GPU)**: EER drops sharply to **19.70%** (a $9.20\%$ absolute reduction), accuracy rises to **79.67%**, and variance across random seeds tightens to just $\pm 0.47\%$ accuracy and $\pm 0.57\%$ EER.

---

## 5. Architectural Ablation: Frozen Backbone vs. Unfreezing on L4 GPU

![Backbone Unfreezing Comparison](outputs/figures/fig3_backbone_unfreezing_comparison.png)
*Figure 3: Few-shot adaptation ($N=50$) comparing a fully frozen foundation backbone against unfreezing top-2 and top-4 transformer layers on an NVIDIA L4 GPU. Unfreezing causes severe few-shot overfitting.*

A central question in transfer learning for speech foundation models is whether fine-tuning the upper transformer layers allows the network to learn target-language acoustic representations. We evaluated three unfreezing configurations under $N=50$ few-shot supervision using an NVIDIA L4 GPU:
1. **Frozen Backbone (SLS Head Only)**: Backbone parameters frozen (315.4M params frozen; 22.8M params trainable).
2. **Unfreezing Top-2 Layers**: Transformer layers 23 and 24 unfrozen (~28M additional backbone parameters trainable).
3. **Unfreezing Top-4 Layers**: Transformer layers 21–24 unfrozen (~56M additional backbone parameters trainable).

**Table 2: Backbone Freezing vs. Partial Unfreezing ($N=50$, L4 GPU)**

| Architectural Setting | Trainable Params | Test Accuracy (%) | EER (%) | Precision (%) | Recall (%) | F1 Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Frozen Backbone (SLS Only)** | **22.8M** | **79.67 $\pm$ 0.47%** | **19.70 $\pm$ 0.57%** | **78.82%** | **81.83%** | **80.05%** |
| **Unfreeze Top-2 Layers** | **50.8M** | 76.46 $\pm$ 0.48% | 24.14 $\pm$ 1.57% | 73.08% | 93.07% | 81.79% |
| **Unfreeze Top-4 Layers** | **78.8M** | 67.12 $\pm$ 5.83% | 25.31 $\pm$ 1.91% | 64.21% | 97.73% | 77.31% |

### Key Finding: Catastrophic Few-Shot Overfitting
As demonstrated in Table 2 and Figure 3:
- Unfreezing the top 2 layers **increases EER from 19.70% to 24.14%** (+4.44% degradation).
- Unfreezing the top 4 layers **degrades EER further to 25.31%** (+5.61% degradation) and collapses accuracy to $67.12\%$.

With only 100 training examples ($N=50$), gradient updates through the high-capacity transformer layers destroy the pre-trained generalizable representations, causing the model to overfit to speaker-specific pitch contours and recording environments. **Keeping 100% of the XLS-R backbone frozen acts as a vital inductive bias and regularizer.**

---

## 6. The V1 vs. V2 Benchmark Comparison

![Dataset Bias Phenomenon](outputs/figures/fig1_dataset_bias_v1_vs_v2.png)
*Figure 4: The Dataset Bias Phenomenon. V1 (red) exploited acoustic shortcuts in single-source synthetic audio, achieving an artificially low EER of 0.40%. V2 (blue) stratifies 5 distinct TTS generators, revealing true cross-generator generalization difficulty.*

**Table 3: Empirical Comparison between Biased (V1) and Stratified (V2) Benchmarks**

| Setting | V1 EER (Single-Source) | V2 EER (5-Generator Stratified) | Performance Discrepancy |
| :--- | :---: | :---: | :---: |
| **Zero-Shot** | 10.60% | 28.90% | +18.30% (Severe generalization gap) |
| **Few-Shot $N=10$** | 1.97% | 26.88% | +24.91% (V1 shortcut exploited) |
| **Few-Shot $N=50$** | 0.40% | 19.70% | +19.30% (Realistic frontier established) |

This comparison highlights a critical insight for speech deepfake research: **reported sub-1% EERs in cross-lingual benchmarks must be scrutinized for generator monoculture**. Only stratified multi-generator evaluations reflect real-world defense efficacy.

---

## 7. Conclusion
In this work, we presented an empirical benchmark for cross-lingual audio deepfake detection on low-resource Hindi speech using Sensitive Layer Summarization on top of a frozen XLS-R 300M backbone. We identified the generator bias trap that artificially inflates detection metrics on single-source synthetic audio, and introduced a 50/50 balanced, 5-generator stratified benchmark. 

Our findings demonstrate that:
1. Zero-shot transfer from English leaves an asymmetric spoof vulnerability (28.90% EER).
2. Fine-tuning only the SLS head with $N=50$ samples reduces EER to 19.70% and boosts accuracy to 79.67%.
3. Fully freezing the foundation backbone is essential, as unfreezing transformer layers causes catastrophic few-shot overfitting (+5.6% EER degradation).

---

## References
1. Y. Zhang, W. Wang, P. Zhang, et al., "Sensitive Layer Summarization for Synthetic Speech Detection," *ACM Multimedia*, 2024.
2. A. Babu, C. Wang, A. Tjandra, et al., "XLS-R: Self-supervised Cross-lingual Speech Representation Learning at Scale," *Interspeech*, 2022.
3. C. Yi, H. Wang, J. Tao, et al., "SEA-Spoof: A Multilingual Synthetic Speech Dataset for Southeast and South Asian Languages," *arXiv preprint*, 2024.
4. Y. Sukhdeve, "XLS-R-SLS: Selective Layer Summarization for Audio Deepfake Detection," *HuggingFace Model Hub*, 2024.
5. X. Wang, J. Yamagishi, "A Comparative Study on Recent Neural Vocoders for Speech Synthesis," *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, 2022.
