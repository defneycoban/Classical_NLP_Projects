# CS114B Spring 2023 Homework 1
# Naive Bayes in Numpy

import os
import numpy as np
from collections import defaultdict

class NaiveBayes():

    def __init__(self):
        self.class_dict = {}
        self.feature_dict = {}
        self.prior = None
        self.likelihood = None

    '''
    Trains a multinomial Naive Bayes classifier on a training set.
    Specifically, fills in self.prior and self.likelihood such that:
    self.prior[class] = log(P(class))
    self.likelihood[feature, class] = log(P(feature|class))
    '''
    def train(self, train_set):
        #initialize counts and class features
        class_counts = defaultdict(int)
        feature_counts = defaultdict(lambda: defaultdict(int))
        doc_count = 0
        vocab_set = set()

        # iterate over training documents
        for root, dirs, files in os.walk(train_set):
            for name in files:
                if name == '.DS_Store':
                    continue

                class_label = os.path.basename(root)
                if class_label not in self.class_dict:
                    self.class_dict[class_label] = len(self.class_dict)
                class_counts[class_label] += 1
                doc_count += 1

                with open(os.path.join(root, name), 'r', encoding='utf-8', errors='ignore') as f:

                    # collect class counts and feature counts
                    document = f.read()
                    tokens = document.split() #PLACEHOLDER FOR TOKENIZATION
                    
                    for token in tokens:
                        token = token.lower()  # Normalize the token
                        vocab_set.add(token)
                        feature_counts[class_label][token] += 1


        # fill in class_dict and feature_dict
        # normalize counts to probabilities, and take logs

        # Assign indices to each feature in the vocab
        self.feature_dict = {feature: idx for idx, feature in enumerate(vocab_set)}
        num_features = len(self.feature_dict)
        num_classes = len(self.class_dict)

        # Initialize prior and likelihood with zeros
        self.prior = np.zeros(num_classes)
        self.likelihood = np.zeros((num_features, num_classes))

        # Convert counts to probabilities
        for class_label, class_index in self.class_dict.items():
            # Calculating prior probabilities
            self.prior[class_index] = np.log(class_counts[class_label] / doc_count)
            
            # Calculating likelihood probabilities
            total_feature_counts = sum(feature_counts[class_label].values())
            for feature, feature_index in self.feature_dict.items():
                # Using Laplace (add 1) smoothing
                self.likelihood[feature_index, class_index] = np.log((feature_counts[class_label][feature] + 1) / (total_feature_counts + num_features))
                    

    '''
    Tests the classifier on a development or test set.
    Returns a dictionary of filenames mapped to their correct and predicted
    classes such that:
    results[filename]['correct'] = correct class
    results[filename]['predicted'] = predicted class
    '''
    def test(self, dev_set):
        results = defaultdict(dict)
        # iterate over testing documents
        for root, dirs, files in os.walk(dev_set):
            for name in files:
                
                if name == '.DS_Store':
                    continue
                
                with open(os.path.join(root, name), 'r', encoding='utf-8', errors='ignore') as f:

                    # create feature vectors for each document
                    document = f.read()
                    tokens = document.split()  # This is a placeholder for actual tokenization
                    feature_vector = np.zeros(len(self.feature_dict))

                    for token in tokens:
                        token = token.lower()
                        if token in self.feature_dict:
                            feature_index = self.feature_dict[token]
                            feature_vector[feature_index] += 1
        
                # get most likely class
                log_posteriors = np.dot(feature_vector, self.likelihood) + self.prior
                predicted_class_index = np.argmax(log_posteriors)
                predicted_class = list(self.class_dict.keys())[predicted_class_index]
                correct_class = os.path.basename(os.path.dirname(os.path.join(root, name)))

                # Store the results
                results[name] = {'correct': correct_class, 'predicted': predicted_class}
        return results

    '''
    Given results, calculates the following:
    Precision, Recall, F1 for each class
    Accuracy overall
    Also, prints evaluation metrics in readable format.
    '''
    def evaluate(self, results):
        # you may find this helpful
        confusion_matrix = np.zeros((len(self.class_dict),len(self.class_dict)))
        
        #print(results)
        for result in results.values():
            correct_class_index = self.class_dict[result['correct']]
            predicted_class_index = self.class_dict[result['predicted']]
            confusion_matrix[correct_class_index, predicted_class_index] += 1

        precision = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=0)
        recall = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)
        f1 = 2 * (precision * recall) / (precision + recall)
        accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)

        # need something to output the results
        for class_index, class_name in enumerate(self.class_dict):
            print(f"Results for class '{class_name}':")
            print(f"Precision: {precision[class_index]}")
            print(f"Recall: {recall[class_index]}")
            print(f"F1 Score: {f1[class_index]}")

        print(confusion_matrix)
        print("Sum of confusion matrix:", np.sum(confusion_matrix))
        print(f"Overall Accuracy: {accuracy}")

if __name__ == '__main__':
    nb = NaiveBayes()
    # make sure these point to the right directories
    # i kept getting an error here with files not being found so i used the absolute path to them 
    nb.train('movie_reviews/train')
    #nb.train('movie_reviews_small/train')
    results = nb.test('movie_reviews/dev')
    #results = nb.test('movie_reviews_small/test')
    nb.evaluate(results)

#print("Current Working Directory:", os.getcwd())
