# CS115B Spring 2024 Homework 2
# Logistic Regression Classifier

import os
from typing import Sequence, DefaultDict, Dict

import numpy as np
from collections import defaultdict
from math import ceil
from random import Random, shuffle
from scipy.special import expit # logistic (sigmoid) function

print(os.getcwd())
class LogisticRegression():

    def __init__(self):
        self.class_dict = {}
        # use of self.feature_dict is optional for this assignment
        self.feature_dict = {}
        self.n_features = None
        self.theta = None # weights (and bias)
        self.last_accuracy = 0  # Initialize the accuracy variable, only for hyperparameter tuning 

    def make_dicts(self, train_set_path: str, n=2) -> None:
        '''
        Given a training set, fills in self.class_dict (and optionally,
        self.feature_dict), as in HW1.
        Also sets the number of features self.n_features and initializes the
        parameter vector self.theta.
        '''
        class_set = set()
        feature_set = set()

        # iterate over training documents
        for root, dirs, files in os.walk(train_set_path):
            if os.path.normpath(train_set_path) == os.path.normpath(root):
                class_set.update(dirs)
                continue  # Skip further processing in this iteration

            for name in files:
                with open(os.path.join(root, name), 'r', encoding='utf-8', errors='ignore') as f:
                    words = f.read().lower().split()
                    feature_set.update(words)
                    # Generate and add bigrams
                    if n >= 2:
                        bigrams = [' '.join(words[i:i+2]) for i in range(len(words)-1)]
                        feature_set.update(bigrams)

        # fill in class_dict, (feature_dict,) n_features, and theta
        # the following are used for testing with the toy corpus from the lab 3
        # exercise
        # Comment this out and replace with your code
        # self.class_dict = {'action': 0, 'comedy': 1}
        # self.feature_dict = {'fast': 0, 'couple': 1, 'shoot': 2, 'fly': 3}
        # self.n_features = 4
        # self.theta = np.zeros(self.n_features + 1)   # +1 for bias
                    
        self.class_dict = {cls: idx for idx, cls in enumerate(sorted(class_set))}
        self.feature_dict = {feature: idx for idx, feature in enumerate(sorted(feature_set))}
        self.n_features = len(self.feature_dict)
        self.theta = np.zeros(self.n_features + 1) # again +1 for bias


    def load_data(self, data_set_path: str):
        '''
        Loads a dataset. Specifically, returns a list of filenames, and dictionaries
        of classes and documents such that:
        classes[filename] = class of the document
        documents[filename] = feature vector for the document (use self.featurize)
        '''
        filenames = []
        classes = dict()
        documents = dict()
        class_counts = {cls: 0 for cls in self.class_dict.keys()}  # Initialize class counts for debuggingg


        # iterate over documents
        for root, dirs, files in os.walk(data_set_path):
            # Skip the top-level directory as it only contains class subdirectories
            # if os.path.normpath(data_set_path) == os.path.normpath(root):
            #     continue

            class_name = os.path.basename(root)

            for name in files:
                if name == '.DS_Store':
                    continue  # Skip those awful .DS_Store files

                full_name = os.path.join(root, name)  # Create full path to file
                filenames.append(full_name)  # Store full path for uniqueness
                classes[full_name] = self.class_dict[class_name]
                class_counts[class_name] += 1  # Increment class count

                with open(full_name, 'r', encoding='utf-8', errors='ignore') as f:
                    words = f.read().split()
                    documents[full_name] = self.featurize(words)       

        return filenames, classes, documents
 

    def featurize(self, document: Sequence[str], n=2) -> np.array:
        '''
        Given a document (as a list of words), returns a feature vector.
        Note that the last element of the vector, corresponding to the bias, is a
        "dummy feature" with value 1.
        '''
        vector = np.zeros(self.n_features + 1)   # + 1 for bias
        # your code here
        document = [word.lower() for word in document]  # Convert each word to lowercase

        for word in document:
            if word in self.feature_dict:  # Check if the word is a recognized feature
                vector[self.feature_dict[word]] += 1  # Increment feature count

        # Count bigrams
        if n >= 2:
            for i in range(len(document)-1):
                bigram = ' '.join(document[i:i+2]).lower()
                if bigram in self.feature_dict:
                    vector[self.feature_dict[bigram]] += 1
        
        vector[-1] = 1   # bias
        return vector


    def train(self, train_set_path: str, batch_size=3, n_epochs=1, eta=0.1) -> None:
        '''
        Trains a logistic regression classifier on a training set.
        '''
        filenames, classes, documents = self.load_data(train_set_path)

        for epoch in range(n_epochs):
            print("Epoch {:} out of {:}".format(epoch + 1, n_epochs))
            loss = 0
            
            for i in range(0, len(filenames), batch_size):
                minibatch_filenames = filenames[i:i + batch_size]
                X = np.array([documents[filename] for filename in minibatch_filenames])
                Y = np.array([classes[filename] for filename in minibatch_filenames])

                # Compute the predicted probabilities (Y_hat) using the logistic (sigmoid) function
                Y_hat = expit(X.dot(self.theta))

                # Clip the predicted probabilities to avoid taking log(0)
                epsilon = 1e-5  # Small value to prevent log(0)
                Y_hat_clipped = np.clip(Y_hat, epsilon, 1 - epsilon)

                # Update the loss using the clipped predicted probabilities
                loss += -np.sum(Y * np.log(Y_hat_clipped) + (1 - Y) * np.log(1 - Y_hat_clipped))

                # Compute the gradient
                gradient = X.T.dot(Y_hat - Y) / len(minibatch_filenames)

                # Update the weights using the gradient and learning rate
                self.theta -= eta * gradient
            
            loss /= len(filenames)
            print("Average Train Loss: {}".format(loss))
            Random(epoch).shuffle(filenames)

            


    def test(self, dev_set_path: str) -> DefaultDict[str, Dict[str, int]]:
        '''
        Tests the classifier on a development or test set.
        Returns a dictionary of filenames mapped to their correct and predicted
        classes such that:
        results[filename]['correct'] = correct class
        results[filename]['predicted'] = predicted class
        '''
        results = defaultdict(dict)
        filenames, classes, documents = self.load_data(dev_set_path)

        for name in filenames:
            # get most likely class (recall that P(y=1|x) = y_hat)
            X = documents[name]  # Feature vector for the document
            Y_hat = expit(X.dot(self.theta))  # Predicted probability
            predicted_class_index = int(Y_hat > 0.5)  # Directly convert the boolean to integer
            predicted_class = [class_name for class_name, index in self.class_dict.items() if index == predicted_class_index][0]
            correct_class_index = classes[name]  # This is the index of the correct class
            correct_class = [class_name for class_name, index in self.class_dict.items() if index == correct_class_index][0]
            results[name] = {'correct': correct_class, 'predicted': predicted_class}

        return results


    def evaluate(self, results: DefaultDict[str, Dict[str, int]]) -> None:
        '''
        Given results, calculates the following:
        Precision, Recall, F1 for each class
        Accuracy overall
        Also, prints evaluation metrics in readable format.
        '''
        # you can copy and paste your code from HW1 here
        confusion_matrix = np.zeros((len(self.class_dict),len(self.class_dict)))
        
        for name, result in results.items():
            correct_class = result['correct']  # This should be a class name
            predicted_class = result['predicted']  # This should also be a class name
            correct_class_index = self.class_dict[correct_class]  # Convert class name to index
            predicted_class_index = self.class_dict[predicted_class]  # Convert class name to index
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
        #print("Sum of confusion matrix:", np.sum(confusion_matrix))
        
         # Set the class variable with the calculated accuracy (FOR HYPERPARAMETER)
        self.last_accuracy = accuracy

        print(f"Overall Accuracy: {accuracy}")


    def find_best_epoch(self, train_set_path: str, validation_set_path: str, max_epochs: int, batch_size: int = 3, eta: float = 0.1):
        best_epoch = 0
        best_accuracy = 0
        self.make_dicts(train_set_path)  # Initialize class and feature dictionaries

        for epoch in range(1, max_epochs + 1):
            print(f"Training for epoch {epoch}...")
            self.train(train_set_path, batch_size=batch_size, n_epochs=1, eta=eta)  # Train for one epoch at a time

            # Evaluate on validation set
            validation_results = self.test(validation_set_path)
            self.evaluate(validation_results)

            # Update best accuracy and epoch if current model is better
            if self.last_accuracy > best_accuracy:
                best_accuracy = self.last_accuracy
                best_epoch = epoch

        print(f"Best Epoch: {best_epoch} with Validation Accuracy: {best_accuracy}")
        return best_epoch, best_accuracy



if __name__ == '__main__':
    lr = LogisticRegression()
    #make sure these point to the right directories

    lr.make_dicts('movie_reviews/train', n=2)
    lr.train('movie_reviews/train', batch_size=3, n_epochs=13, eta=0.05)
    results = lr.test('movie_reviews/dev')
    lr.evaluate(results)

    #This is the code I used to find the best epoch
    #I commented it out because it only needed to be run once 
    #best_epoch, best_accuracy = lr.find_best_epoch('movie_reviews/train', 'movie_reviews/dev', max_epochs=20)

    # lr.make_dicts('movie_reviews_small/train')
    # lr.train('movie_reviews_small/train', batch_size=3, n_epochs=1, eta=0.1)
    # results = lr.test('movie_reviews_small/test')
    # lr.evaluate(results)