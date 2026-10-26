# Decoupled Evidential Distillation with Variational Quantum Circuits for Adversarial Robustness in Financial Sentiment Classification

## Description

This repository contains the training code for the DED and VQC-DED defences
and the hyperparameter configuration used in the paper.

Financial sentiment classifiers on short, noisy social-media text are highly
sensitive to small adversarial edits. Decoupled Evidential Distillation (DED)
addresses this by (i) splitting the classification head into a Mean Branch
that carries the class probabilities `m` and a gradient-detached Precision
Branch that carries the scalar evidence `S`, (ii) distilling both signals from a
frozen teacher on clean text, and (iii) aggregating predictions at inference
with evidence-weighted smoothing (`alpha-smooth`) over N=100 masked input variants
with masking probability rho=0.15. VQC-DED replaces the linear
Mean Branch with a variational quantum circuit on n=5 qubits and L=4 layers,
while keeping the Precision Branch, the Dirichlet parameterisation
`alpha = S * m` and the alpha-smooth inference rule unchanged.

## Dataset Information

The study evaluates DED and VQC-DED on four publicly available financial tweet
datasets for three-way sentiment classification (negative, neutral, positive).
All four are accessed through the consolidated Hugging Face collection
`TimKoornstra/financial-tweets-sentiment` [Koornstra, 2023]:

- URL: <https://huggingface.co/datasets/TimKoornstra/financial-tweets-sentiment>
- Revision: `bba602698fd6f18226f6404fd8fb637ef67fa1eb`

Each subset is recovered from the collection by its `url` field. Labels
bearish, neutral and bullish map to negative (0), neutral (1) and positive (2).

| Subset (`--dataset`) | Total | Pos | Neu | Neg | Train | Test |
|---|---|---|---|---|---|---|
| `master_thesis` | 8,620 | 3,703 | 2,757 | 2,160 | 6,465 | 2,155 |
| `stock_market_tweets` | 1,280 | 521 | 418 | 341 | 960 | 320 |
| `stockrel` | 6,153 | 4,810 | 610 | 733 | 4,614 | 1,539 |
| `twitter_stock_sentiment` | 1,747 | 699 | 455 | 593 | 1,310 | 437 |

Upstream sources:

1. **Master Thesis Data** [Wilksch and Abramova, 2023]:
   <https://github.com/moritzwilksch/MasterThesis/tree/main>,
   publication DOI <https://doi.org/10.1016/j.jjimei.2023.100171>.
2. **Stock Market Tweets Data** [Koornstra, 2023]:
   <https://ieee-dataport.org/open-access/stock-market-tweets-data>,
   DOI <https://doi.org/10.21227/g8vy-5w61>.
3. **Stock Related Tweet Sentiment** (StockRel) [Gilgo, 2021]:
   <https://www.kaggle.com/datasets/mattgilgo/stock-related-tweet-sentiment>.
4. **Twitter Stock Sentiment** [Thakoor, 2019]:
   <https://github.com/poojathakoor/twitter-stock-sentiment>.

**Split policy.** Empty texts and exact duplicates are removed first. Each
dataset is then split once into 75% training and 25% test data by stratified
sampling on the label (seed 42). Validation data for early stopping are drawn
from the training partition only; the test partition is not used for model
selection.

## Code Information

```
configs/default.json        Training, DED loss, VQC and inference settings.
ded/data.py                 Text normalisation, vocabulary, tokenisers, datasets.
ded/models.py               Backbones, DED head, VQC Mean Branch, control heads, DED loss.
ded/quantum.py              State-vector simulation of the VQC.
ded/inference.py            Inference rules: alpha-smooth, equal-weight smoothing, single pass.
ded/training.py             Shared training loop.
scripts/prepare_data.py     Entry point: download and preprocess the four datasets.
scripts/train_teacher.py    Entry point: train the undefended teachers.
scripts/train_student.py    Entry point: train DED, VQC-DED and ablation students.
scripts/evaluate_clean.py   Entry point: clean accuracy, Macro-F1 and class-wise F1.
```

## Usage Instructions

Commands assume the repository root as the working directory.

### 1. Preprocess the datasets

```
python scripts/prepare_data.py --out data
```

### 2. Train the teachers

```
python scripts/train_teacher.py --dataset stockrel --kind bigru
python scripts/train_teacher.py --dataset stockrel --kind finbert
```

`--kind` is one of `bilstm`, `bigru`, `han` (full-width teachers) or
`finbert` (teacher of the DistilBERT student).

### 3. Train DED and VQC-DED students

```
python scripts/train_student.py --dataset stockrel --kind bigru --variant ded
python scripts/train_student.py --dataset stockrel --kind bigru --variant vqc_ded
python scripts/train_student.py --dataset stockrel --kind distilbert --variant ded
```

Ablation students are selected with `--variant` (for example `kd`,
`no_detach`, `coupled`, `no_evi`, `vqc_prec`, `matched_mlp`), the distillation
temperature with `--temperature`, and the circuit size with `--qubits` and
`--layers`.

### 4. Clean evaluation

```
python scripts/evaluate_clean.py --dataset stockrel --kind bigru --run vqc_ded
python scripts/evaluate_clean.py --dataset stockrel --kind bigru --run ded --rule single
```

`--rule` selects `alpha` (default), `uniform` (equal weights, m-smooth) or `single` inference.

## Requirements

```
pip install -r requirements.txt
```

- `torch` (2.6.0 used in the paper)
- `transformers`, `datasets`
- `numpy`, `pandas`, `scikit-learn`

**Computing infrastructure.** All experiments in the paper were run on a single
NVIDIA GeForce RTX 3060 GPU (12 GB VRAM) under Windows 10 IoT Enterprise LTSC
2021 (build 10.0.19044), with Python 3.13.5, PyTorch 2.6.0 and CUDA 12.4.

## Methodology

**Preprocessing.** Texts are lowercased, URLs and user mentions are replaced
with `[URL]` and `[USER]`, and whitespace is normalised; cashtags, hashtags,
stopwords and spelling are kept. The vocabulary for recurrent encoders (at
most 10,000 tokens) is built from the training partition, and the maximum
input length is 128 tokens.

**Training objective.** A frozen teacher produces soft targets
`pt = softmax(z_t / T)` with T=5. The student's Decoupled Evidential Head
outputs `m = softmax(W_m h)` and `S = Softplus(W_s detach(h)) + 1`, giving
`alpha = S * m`. A projector maps `pt` to a teacher evidence `St`, so that
`alpha_t = St * pt`. The loss is

```
L = L_kl + 0.5 L_mean + L_shape + 0.1 L_evi + 0.001 L_soft
```

Models are trained with AdamW (weight decay 0.0005), at most 30 epochs and
early stopping with patience 5 on validation Macro-F1.

**Inference (alpha-smooth).** Each attended token is masked independently with
probability rho=0.15; the N=100 masked variants give `(m_i, S_i)` and

```
m_bar = sum_i S_i m_i / sum_j S_j,    y_hat = argmax_k m_bar[k]
```

**VQC Mean Branch.** A linear projection maps `h` to five angles `pi * h'`;
each of four layers applies the `R_Y` encoding, trainable `R_Y` and `R_Z`
rotations and a CNOT ring; Pauli-Z expectations are mapped to class scores by
a linear readout.

**Adversarial attack protocol.** The paper uses a black-box,
query-based saliency-driven protocol that combines character-level edits
[Gao et al., 2018], PWWS-style synonym substitution [Ren et al., 2019] and
sentence-level paraphrase with DeepSeek-V3.2 [Madrueno et al., 2025], with a
budget of three edits per tweet, five rounds and 100 correctly classified
originals per round, regenerated for every defence and backbone.

## Citations

If you use this code, please cite:

- Ling, J. and Gan, K. H. Decoupled evidential distillation with variational
  quantum circuits for adversarial robustness in financial sentiment
  classification. *PeerJ Computer Science* (under review).

Datasets:

- Koornstra, T. (2023). *Financial Tweets Sentiment Dataset.* Hugging Face.
  <https://huggingface.co/datasets/TimKoornstra/financial-tweets-sentiment>
- Wilksch, M. and Abramova, O. (2023). PyFin-Sentiment: Towards a
  machine-learning-based model for deriving sentiment from financial tweets.
  *International Journal of Information Management Data Insights*, 3(1),
  100171. <https://doi.org/10.1016/j.jjimei.2023.100171>
- Gilgo, M. (2021). *Stock Related Tweet Sentiment Dataset.* Kaggle.
- Thakoor, P. (2019). *Twitter Stock Sentiment.* GitHub.

Models and methods:

- Araci, D. (2019). FinBERT: Financial sentiment analysis with pre-trained
  language models. arXiv:1908.10063.
- Sanh, V., Debut, L., Chaumond, J. and Wolf, T. (2019). DistilBERT, a
  distilled version of BERT. arXiv:1910.01108.
- Gao, J., Lanchantin, J., Soffa, M. L. and Qi, Y. (2018). Black-box generation
  of adversarial text sequences to evade deep learning classifiers. *IEEE
  Security and Privacy Workshops*, 50-56.
- Ren, S., Deng, Y., He, K. and Che, W. (2019). Generating natural language
  adversarial examples through probability weighted word saliency. *ACL*,
  1085-1097.
- Madrueno, N., Fernandez-Isabel, A., Fernandez, R. R. and de Diego, I. M.
  (2025). Advancing text adversarial example generation using large language
  models. *Knowledge-Based Systems*, 329, 114361.

## Licence and Contribution Guidelines

The code is released under the MIT License (see `LICENSE`). The datasets
remain under the licences of their original providers. Issues and pull
requests are welcome.
