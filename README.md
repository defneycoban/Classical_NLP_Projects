# Classical NLP Projects

## Overview
This repository contains various projects that explore classical Natural Language Processing (NLP) methods. Each project demonstrates different aspects of traditional NLP techniques and their applications. The projects included are Naive Bayes Sentiment Analysis, Logistic Regression Sentiment Analysis, Structured Perceptron for POS Tagging, and Distributional Semantics and Word Embeddings.

## Projects

### 1. Naive Bayes Sentiment Analysis
**Description**: This project implements a Naive Bayes classifier for sentiment analysis on movie reviews. The goal is to classify movie reviews as either positive or negative based on their content.

**Features**:
- Data preparation and tokenization
- Naive Bayes classifier implementation
- Model training and evaluation

**Usage**:
1. Clone the repository:
    ```sh
    git clone <repository-url>
    cd classical-nlp-projects/naive-bayes-sentiment-analysis
    ```
2. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```
3. Run the classifier:
    ```sh
    python naive_bayes.py
    ```

### 2. Logistic Regression Sentiment Analysis
**Description**: This project implements a logistic regression classifier for sentiment analysis on movie reviews. The aim is to classify movie reviews as either positive or negative by leveraging various features and optimizing the classifier's performance through hyperparameter tuning.

**Features**:
- Data preparation and feature extraction
- Logistic regression classifier implementation
- Model training, evaluation, and hyperparameter tuning

**Usage**:
1. Clone the repository:
    ```sh
    git clone <repository-url>
    cd classical-nlp-projects/logistic-regression-sentiment-analysis
    ```
2. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```
3. Run the classifier:
    ```sh
    python logistic_regression.py
    ```

### 3. Structured Perceptron for POS Tagging
**Description**: This project implements a structured perceptron to perform part-of-speech (POS) tagging. The goal is to accurately tag words in sentences with their corresponding parts of speech using a structured learning approach.

**Features**:
- Data preparation and dictionary creation
- Structured perceptron implementation with Viterbi algorithm
- Model training and evaluation

**Usage**:
1. Clone the repository:
    ```sh
    git clone <repository-url>
    cd classical-nlp-projects/structured-perceptron-pos-tagger
    ```
2. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```
3. Run the POS tagger:
    ```sh
    python pos_tagger.py
    ```

### 4. Distributional Semantics and Word Embeddings
**Description**: This project explores the creation and application of distributional semantic word vectors. The goal is to develop semantic representations of words from a corpus and utilize these representations in various computational lexical semantic tasks, such as synonym detection and analogy resolution.

**Features**:
- Co-occurrence matrix computation and PPMI transformation
- Dimensionality reduction using SVD
- Similarity computation and synonym detection
- Solving SAT analogy questions using word vectors

**Usage**:
1. Clone the repository:
    ```sh
    git clone <repository-url>
    cd classical-nlp-projects/distributional-semantics
    ```
2. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```
3. Run the analysis:
    ```sh
    python main.py
    ```
