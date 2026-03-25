# Assignment 2 – NLP and Sequence Models

This repository contains my implementation and analysis for Assignment 2.  
The assignment focuses on two main tasks:

1. Training Word2Vec models (CBOW and Skip-Gram) on a corpus and analyzing the learned embeddings.
2. Building character-level neural networks to generate names using different architectures.

The repository includes both the implementation and the report explaining the results.

---

## Repository Structure

```
.
├── M25MAC012_A2_Prob1.ipynb   # Word2Vec training, analysis, PCA visualization
├── M25MAC012_prob2.py         # Character-level name generation models
├── M25MAC012-A2_report.pdf    # Assignment report
└── TrainingNames.txt          # Dataset required for Problem 2
```

---

## Problem 1 – Word Embeddings

In this part of the assignment, Word2Vec models were trained on a corpus created from academic and institutional content.

After preprocessing:
- Total documents: 1  
- Total tokens: 61,726  
- Vocabulary size: 3,479  

Two Word2Vec models were trained:

### CBOW Model
- Embedding dimension: 300
- Window size: 5
- Negative samples: 10
- Epochs: 10
- Learning rate: 0.01

### Skip-Gram Model
- Embedding dimension: 300
- Window size: 5
- Negative samples: 10
- Epochs: 10
- Learning rate: 0.01

Observations from the analysis:
- CBOW tends to average context and sometimes loses specific relationships.
- Skip-Gram performs better at capturing meaningful relationships and rare words.

The notebook includes:
- Word cloud visualization
- Word similarity analysis
- Analogy tasks
- PCA visualization of embeddings

---

## Problem 2 – Name Generation using RNNs

This script implements a character-level neural network that learns to generate names.

Three models are implemented using PyTorch:
- Vanilla RNN
- Bidirectional LSTM (BLSTM)
- RNN with Attention

### Model Configurations

#### Vanilla RNN
- Hidden size: 64
- Layers: 1
- Learning rate: 0.003
- Epochs: 20
- Trainable parameters: 10,868

#### BLSTM
- Hidden size: 64
- Layers: 1 (bidirectional)
- Learning rate: 0.003
- Epochs: 10
- Trainable parameters: 66,612

#### RNN with Attention
- Hidden size: 64
- Layers: 1
- Learning rate: 0.003
- Epochs: 20
- Trainable parameters: 17,576

### Evaluation Metrics

The generated names are evaluated using:
- Novelty Rate
- Diversity

Summary of results:
- Vanilla RNN produced many new names with good diversity.
- Attention-based RNN also performed well and generated more structured names.
- BLSTM generated several unrealistic names compared to the other models.

---

# How to Run the Project

## 1. Clone the Repository

```
git clone https://github.com/your-username/assignment-2-nlp.git
cd assignment-2-nlp
```

---

## 2. Install Dependencies

Make sure Python 3.8+ is installed.

Install the required libraries:

```
pip install torch numpy matplotlib scikit-learn jupyter
```

---

## 3. Run Problem 1 (Word2Vec Notebook)

Open the notebook:

```
jupyter notebook M25MAC012_A2_Prob1.ipynb
```

Run all cells to:
- Train embeddings
- Visualize word clusters
- Perform similarity and analogy analysis

---

## 4. Run Problem 2 (Name Generation Model)

Make sure the dataset file exists in the same folder:

```
TrainingNames.txt
```

Then run:

```
python M25MAC012_prob2.py
```

During training, the script will:
- Train three different models
- Display loss values
- Generate sample names
- Compute novelty and diversity scores
- Save trained models as `.pth` files

---

## Output

After training, the script generates names such as:

```
Harshana
Kavirata
Mirita
```

Some unrealistic names may also appear, which is expected in character-level generation models.

---

## Implementation Notes

A few important details about the implementation:
- Characters are encoded using one-hot vectors.
- Gradient clipping is used during training for stability.
- Temperature sampling is applied during generation to improve diversity.
- The attention model keeps track of previous hidden states while generating names.

---

## Author

Smithi Sureshan  
M25MAC012
