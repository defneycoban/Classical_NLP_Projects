# CS115B Spring 2024 Homework 3
# Part-of-speech Tagging with Structured Perceptrons

import os
import pickle

import numpy as np
from collections import defaultdict
from random import Random

class POSTagger(): 

    def __init__(self):
        # For testing with the toy corpus from the lab 7 exercise
        self.tag_dict = {}
        self.word_dict = {}
        self.initial = np.array([-0.3, -0.7, 0.3])
        self.transition = np.array([[-0.7, 0.3, -0.3],
                                    [-0.3, -0.7, 0.3],
                                    [0.3, -0.3, -0.7]])
        self.emission = np.array([[-0.3, -0.7, 0.3],
                                  [0.3, -0.3, -0.7],
                                  [-0.3, 0.3, -0.7],
                                  [-0.7, -0.3, 0.3],
                                  [0.3, -0.7, -0.3],
                                  [-0.7, 0.3, -0.3]])
        
        # Should raise an IndexError; if you come across an unknown word, you
        # Should treat the emission scores for that word as 0
        
        # Tried using this as np.inf at first but it gave me problems so I am now changing it to None and Adding UNK to word_dict
        self.unk_index = None

    def make_dicts(self, train_set):
        '''
        Fills in self.tag_dict and self.word_dict, based on the training data.
        '''
        tag_index = 0
        word_index = 0

        # Iterate over training documents
        for root, dirs, files in os.walk(train_set):
            for name in files:

                # file_path = os.path.join(root, name)
                # print(f"Reading file: {file_path}")

                with open(os.path.join(root, name), 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line:  # Skip empty lines
                            continue
                        
                        # Split the line into word/tag pairs
                        pairs = line.split()  # Splits the line by whitespace
                        for pair in pairs:
                            # Split each pair by the last '/'
                            parts = pair.rsplit('/', 1)
                            if len(parts) != 2:
                                continue  # Skip if the format is incorrect
                            word, tag = parts[0], parts[1]
                            if tag not in self.tag_dict:
                                self.tag_dict[tag] = tag_index
                                tag_index += 1
                            if word not in self.word_dict:
                                self.word_dict[word] = word_index
                                word_index += 1

         # Add <UNK> to word_dict with a unique index
        self.word_dict['<UNK>'] = len(self.word_dict)
        self.tag_dict['<UNK>'] = len(self.tag_dict)
        self.unk_index = self.word_dict['<UNK>']

    def load_data(self, data_set):
        '''
        Loads a dataset. Specifically, returns a list of sentence_ids, and
        dictionaries of tag_lists and word_lists such that:
        tag_lists[sentence_id] = list of part-of-speech tags in the sentence
        word_lists[sentence_id] = list of words in the sentence
        '''
        sentence_ids = []
        tag_lists = {}
        word_lists = {}
        sentence_id = 0

        # Iterate over documents
        for root, dirs, files in os.walk(data_set):
            for name in files:
                with open(os.path.join(root, name), 'r') as f:
                    current_tags = []
                    current_words = []
                    for line in f:
                        line = line.strip()
                        if not line:  # Skip empty lines
                            if current_words:  # Only add non-empty sentences
                                sentence_ids.append(sentence_id)
                                tag_lists[sentence_id] = current_tags
                                word_lists[sentence_id] = current_words
                                sentence_id += 1
                                current_tags = []
                                current_words = []
                            continue  # Move to the next line
                        
                        # Processing logic for non-empty lines
                        pairs = line.split()
                        for pair in pairs:
                            word, tag = pair.rsplit('/', 1)
                            word_idx = self.word_dict.get(word, self.word_dict['<UNK>'])
                            tag_idx = self.tag_dict.get(tag, self.tag_dict['<UNK>'])
                            current_words.append(word_idx)
                            current_tags.append(tag_idx)

                    # Check if the last sentence in the file was processed
                    if current_words:  # Add the last sentence if the file doesn't end with an empty line
                        sentence_ids.append(sentence_id)
                        tag_lists[sentence_id] = current_tags
                        word_lists[sentence_id] = current_words

        return sentence_ids, tag_lists, word_lists



    def viterbi(self, sentence):
        '''
        Implements the Viterbi algorithm.
        Use v and backpointer to find the best_path.
        '''
        T = len(sentence)  # Length of the sentence
        N = len(self.tag_dict)  # Number of tags

        # Initialize matrices
        v = np.full((N, T), -np.inf)  # Use negative infinity for initial scores
        backpointer = np.zeros((N, T), dtype=int)

        # Convert sentence words to indices, handling unknown words
        word_indices = [self.word_dict.get(word, self.unk_index) for word in sentence]

        # Initialization step
        v[:, 0] = self.initial + self.emission[word_indices[0], :]

        # Recursion step
        for t in range(1, T):
            word_index = word_indices[t]
            # Vectorized update for all states and backpointers at once
            scores = v[:, t-1].reshape(-1, 1) + self.transition + self.emission[word_index, :]
            v[:, t] = np.max(scores, axis=0)
            backpointer[:, t] = np.argmax(scores, axis=0)

        # Termination
        last_tag = np.argmax(v[:, T-1])
        best_path = [last_tag]

        for t in range(T-1, 0, -1):
            best_path.insert(0, backpointer[best_path[0], t])

        # Convert tag indices to tag strings
        inv_tag_dict = {v: k for k, v in self.tag_dict.items()}
        best_path_tags = [inv_tag_dict[tag] for tag in best_path]

        return best_path_tags


    def train(self, train_set):
        '''
        Trains a structured perceptron part-of-speech tagger on a training set.
        '''
        self.make_dicts(train_set)
        sentence_ids, tag_lists, word_lists = self.load_data(train_set)
        Random(0).shuffle(sentence_ids)

        self.initial = np.zeros(len(self.tag_dict))
        self.transition = np.zeros((len(self.tag_dict), len(self.tag_dict)))
        self.emission = np.zeros((len(self.word_dict), len(self.tag_dict)))

        # For debugging: print the size of the dataset and the number of unique tags and words
        # print(self.word_dict)
        # print(f"Training on {len(sentence_ids)} sentences with {len(self.tag_dict)} unique tags and {len(self.word_dict)-1} unique words.")

        # Reverse lookup for tag dictionary
        tag_lookup = {v: k for k, v in self.tag_dict.items()}

        for i, sentence_id in enumerate(sentence_ids):
            words = word_lists[sentence_id]
            correct_tags_indices = tag_lists[sentence_id]
            predicted_tags = self.viterbi(words)
            correct_tags = [tag_lookup[idx] for idx in correct_tags_indices]

            # reinforce the correct initial state
            self.initial[self.tag_dict[correct_tags[0]]] += 1

            if correct_tags[0] != predicted_tags[0]:
            # Penalize the incorrect prediction only if it's wrong
                self.initial[self.tag_dict[predicted_tags[0]]] -= 1
       
            for t, word_idx in enumerate(words):
                correct_tag = correct_tags[t]
                predicted_tag = predicted_tags[t]

                # Update emission weights
                self.emission[word_idx, self.tag_dict[correct_tag]] += 1
                if correct_tag != predicted_tag:
                # Only penalize if the prediction was incorrect
                    self.emission[word_idx, self.tag_dict[predicted_tag]] -= 1

                if t > 0:
                    correct_prev_tag = correct_tags[t-1]
                    predicted_prev_tag = predicted_tags[t-1]

                     # Update transition weights
                    self.transition[self.tag_dict[correct_prev_tag], self.tag_dict[correct_tag]] += 1
                    if correct_prev_tag != predicted_prev_tag or correct_tag != predicted_tag:
                        # Penalize transitions for incorrect sequences
                        self.transition[self.tag_dict[predicted_prev_tag], self.tag_dict[predicted_tag]] -= 1

            # Prints progress of training
            if (i + 1) % 1000 == 0 or i + 1 == len(sentence_ids):
                print(i + 1, 'training sentences tagged')
    
        print("Training completed.")


    def test(self, dev_set):
        '''
        Tests the tagger on a development or test set.
        Returns a dictionary of sentence_ids mapped to their correct and predicted
        sequences of part-of-speech tags such that:
        results[sentence_id]['correct'] = correct sequence of tags
        results[sentence_id]['predicted'] = predicted sequence of tags
        '''
        results = defaultdict(dict)
        sentence_ids, tag_lists, word_lists = self.load_data(dev_set)

        # Reverse lookup for tag dictionary, to convert indices back to strings if needed
        inv_tag_dict = {v: k for k, v in self.tag_dict.items()}

        for i, sentence_id in enumerate(sentence_ids):
            # Convert sentence words to indices, handling unknown words
            words = word_lists[sentence_id]
            predicted_tags = self.viterbi(words)  # Directly use the tags from Viterbi as strings

            # Prepare a list for correct tags, checking for missing indices
            correct_tags = []
            for idx in tag_lists[sentence_id]:
                if idx not in inv_tag_dict:
                    print(f"Warning: Missing tag index {idx} in sentence_id {sentence_id}. Skipping this tag.")
                    continue  # Skip this tag
                correct_tags.append(inv_tag_dict[idx])

            results[sentence_id]['correct'] = correct_tags
            results[sentence_id]['predicted'] = predicted_tags

            if (i + 1) % 1000 == 0 or i + 1 == len(sentence_ids):
                print(i + 1, 'testing sentences tagged')
        return results


    def evaluate(self, results):
        '''
        Given results, calculates overall accuracy.
        '''
        total_tags = 0
        correct_tags = 0
        for sentence_id in results:
            total_tags += len(results[sentence_id]['correct'])
            correct_tags += sum(1 for i, tag in enumerate(results[sentence_id]['correct']) if tag == results[sentence_id]['predicted'][i])

        accuracy = correct_tags / total_tags if total_tags > 0 else 0
        return accuracy




if __name__ == '__main__':
    pos = POSTagger()

    # Small datasets
    pos.train('data_small/train')
    results = pos.test('data_small/test')

    # # Full dataset

    # pos.train('brown/train')
    # #Writes the POS tagger to a file
    # with open('pos_tagger.pkl', 'wb') as f:
    #     pickle.dump(pos, f)
    # # Reads the POS tagger from a file
    # with open('pos_tagger.pkl', 'rb') as f:
    #     pos = pickle.load(f)
    # results = pos.test('brown/dev')

    print('Accuracy:', pos.evaluate(results))
