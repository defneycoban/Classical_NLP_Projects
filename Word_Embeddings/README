# Distributional Semantics and Word Embeddings

## Overview
This project explores the creation and application of distributional semantic word vectors. The goal is to develop semantic representations of words from a corpus and utilize these representations in various computational lexical semantic tasks, such as synonym detection and analogy resolution.

## Features
- **Co-occurrence Matrix**: Construct a co-occurrence matrix from a given corpus.
- **PPMI Transformation**: Apply Positive Pointwise Mutual Information (PPMI) to the co-occurrence matrix.
- **Dimensionality Reduction**: Reduce the dimensionality of word vectors using Singular Value Decomposition (SVD).
- **Similarity Computation**: Calculate Euclidean distances between word pairs to analyze semantic similarity.
- **Synonym Detection**: Implement methods to detect synonyms using distributional semantics and pre-trained word vectors.
- **SAT Analogy Questions**: Solve SAT analogy questions using word vectors.

## Dependencies
- Python
- Numpy
- Scipy

## Implementation Details

### 1. Creating Distributional Semantic Word Vectors
1. **Compute Co-occurrence Matrix**: Calculate the co-occurrence matrix \(C\) from the corpus, where \(C[w, c]\) represents the number of bigrams \((w, c)\) and \((c, w)\) in the corpus.
2. **Apply PPMI Transformation**: Transform the co-occurrence matrix using Positive Pointwise Mutual Information (PPMI).
3. **Dimensionality Reduction**: Use Singular Value Decomposition (SVD) to reduce the dimensionality of the PPMI matrix.
4. **Vector Comparison**: Compute Euclidean distances between word pairs to evaluate the semantic relationships.

### 2. Computing with Distributional Semantic Word Vectors
1. **Synonym Detection**:
    - Create a multiple-choice synonym test set.
    - Use word vectors to detect synonyms based on Euclidean distance and cosine similarity.
    - Compare results using classical distributional semantic vectors and Google's word2vec vectors.
2. **SAT Analogy Questions**:
    - Use word vectors to solve SAT analogy questions.
    - Experiment with different methods for creating word vectors for the analogy task.

## Usage
1. **Clone the Repository**:
    ```sh
    git clone <repository-url>
    cd distributional-semantics
    ```
2. **Install Dependencies**:
    ```sh
    pip install -r requirements.txt
    ```
3. **Run the Analysis**:
    ```sh
    python main.py
    ```

## Results
The project successfully demonstrates the creation of distributional semantic models and their application to lexical semantic tasks. The PPMI transformation and SVD reduction significantly improve the quality of the word vectors.

### Synonym Detection
The results of the synonym detection task, compared using Euclidean distance and cosine similarity, show that the word2vec vectors perform better than the classical distributional semantic vectors in detecting synonyms.

### SAT Analogy Questions
The word vectors achieve an accuracy of around 30% on SAT analogy questions, demonstrating the effectiveness of the models in capturing semantic relationships.

## Conclusion
This project provides a comprehensive exploration of distributional semantics and word embeddings. By developing and applying these models to real-world tasks, I gained valuable insights into the strengths and limitations of different approaches in natural language processing.
