Assignment 2 – NLP and Sequence Models

This repository contains my implementation and analysis for Assignment 2.
The assignment focuses on two main tasks:

Training Word2Vec models (CBOW and Skip-Gram) on a corpus and analyzing embeddings.
Building character-level neural networks to generate names using different architectures:
Vanilla RNN
Bidirectional LSTM (BLSTM)
RNN with Attention

The report with detailed explanations, results, and visualizations is also included in this repository.

Repository Structure
.
├── M25MAC012_A2_Prob1.ipynb   # Word2Vec training, analysis, PCA visualization
├── M25MAC012_prob2.py         # Character-level name generation models
├── M25MAC012-A2_report.pdf    # Assignment report
└── TrainingNames.txt          # Dataset used for training (required to run Prob2)
Problem 1 – Word Embeddings

In this part of the assignment, Word2Vec models were trained on a corpus built from academic and institutional content.

Dataset summary from the preprocessing step:

Total documents: 1
Total tokens: 61,726
Vocabulary size: 3,479

Two models were trained:

CBOW
Skip-Gram

Both used:

Embedding dimension: 300
Window size: 5
Negative samples: 10
Epochs: 10
Learning rate: 0.01

From the observations:

CBOW struggled with capturing strong semantic relationships.
Skip-Gram performed better, especially with rare words and meaningful word associations.

The notebook includes:

Word cloud visualization
Nearest neighbor comparisons
Word analogies
PCA visualization of embeddings
Problem 2 – Name Generation using RNNs

This script implements a character-level language model that learns to generate new names.

Three models are implemented from scratch in PyTorch:

Vanilla RNN
Bidirectional LSTM
RNN with Attention

The models are trained on a dataset of names and then used to generate new ones.

Model Configurations

Vanilla RNN

Hidden size: 64
Layers: 1
Learning rate: 0.003
Epochs: 20
Parameters: 10,868

BLSTM

Hidden size: 64
Layers: 1 (bidirectional)
Learning rate: 0.003
Epochs: 10
Parameters: 66,612

RNN with Attention

Hidden size: 64
Layers: 1
Learning rate: 0.003
Epochs: 20
Parameters: 17,576

The evaluation is done using:

Novelty Rate
Diversity

Results observed:

Vanilla RNN and Attention model produced more realistic names.
BLSTM generated many unrealistic outputs.
How to Run the Code
1. Clone the Repository
git clone https://github.com/your-username/assignment-2-nlp.git
cd assignment-2-nlp
2. Install Requirements

Make sure you have Python 3.8+ installed.

Install dependencies:

pip install torch numpy matplotlib scikit-learn

If you are running the notebook:

pip install jupyter
3. Run Problem 1 (Word2Vec)

Open the notebook:

jupyter notebook M25MAC012_A2_Prob1.ipynb

Then run all cells to:

Train embeddings
Visualize results
Perform analysis
4. Run Problem 2 (Name Generator)

Make sure this file exists in the project folder:

TrainingNames.txt

Then run:

python M25MAC012_prob2.py

During training you will see output like:

Training VanillaRNN | Parameters: ...
Epoch 1, Loss: ...
Epoch 10, Loss: ...
...

After training:

The script generates sample names
Calculates novelty and diversity
Saves model weights (.pth files)
Output

The script will:

Train all three models
Generate 1000 names per model
Print evaluation metrics
Save trained models

Example outputs include names such as:

Harshana
Kavirata
Mirita
...

(Some unrealistic names are also generated, which is expected for character models.)

Notes

A few implementation details:

One-hot character encoding is used as input.
Gradient clipping is applied during training for stability.
Temperature sampling is used during generation to improve diversity.
Attention model keeps track of previous hidden states while generating names.
Author

Smithi Sureshan
M25MAC012
