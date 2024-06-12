# Structured Perceptron for POS Tagging

## Overview
This project implements a structured perceptron to perform part-of-speech (POS) tagging. The goal is to accurately tag words in sentences with their corresponding parts of speech using a structured learning approach. The implementation leverages the Viterbi algorithm for efficient sequence prediction.

## Dataset
The project uses the following datasets:
- **Brown Corpus**: A collection of POS-tagged sentences from the Brown corpus, divided into training (80%), development (10%), and testing (10%) sets.
- **Toy Corpus from Jurafsky and Martin**: A smaller dataset used for initial testing and experimentation, described in the book "Speech and Language Processing" by Jurafsky and Martin.

## Features
- **Data Preparation**: Load and preprocess the POS-tagged sentences into appropriate data structures.
- **Structured Perceptron**: Implement a structured perceptron algorithm for sequence prediction.
- **Viterbi Algorithm**: Efficiently predict the best sequence of tags for a given sentence using the Viterbi algorithm.
- **Model Training and Evaluation**: Train the structured perceptron model and evaluate its performance on the development set.

## Dependencies
- Python
- Numpy

## Implementation Details
The implementation involves the following steps:
1. **Data Loading**: Load and preprocess the POS-tagged sentences from the provided datasets.
2. **Dictionary Creation**: Create dictionaries to translate between indices and words/tags.
3. **Feature Extraction**: Implement the Viterbi algorithm to generate trellises for sequence prediction.
4. **Model Training**: Train the structured perceptron model using the training set.
5. **Evaluation**: Evaluate the model's performance on the development set, calculating overall accuracy.

## Usage
1. **Clone the Repository**:
    ```sh
    git clone <repository-url>
    cd structured-perceptron-pos-tagger
    ```
2. **Install Dependencies**:
    ```sh
    pip install -r requirements.txt
    ```
3. **Run the POS Tagger**:
    ```sh
    python pos_tagger.py
    ```

## Results
The structured perceptron model achieves high accuracy on the POS tagging task, demonstrating the effectiveness of structured learning and the Viterbi algorithm for sequence prediction.

## Conclusion
This project provides a hands-on implementation of a structured perceptron for POS tagging. By working with real-world datasets and implementing efficient algorithms, I gained valuable insights into the application of structured learning techniques in natural language processing.
