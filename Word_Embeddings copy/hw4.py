import random
from random import sample, choice
import numpy as np
import scipy.linalg as scipy_linalg
from scipy.spatial.distance import cosine
import os
import json

random.seed(42)
np.random.seed(42)

# Helper function to compute Euclidean distance
def euclidean_distance(vec1, vec2):
    return scipy_linalg.norm(vec1 - vec2)

def cosine_similarity(vec1, vec2):
    return 1 - cosine(vec1, vec2)

def part1():
    """
    Runs part one of Homework 4.

    Creates the co-occurrence matrix.
    Prints the co-occurrence matrix.

    Creates the ppmi matrix.
    Prints the ppmi matrix.

    Evaluate word similarity with different distance metrics for each word in the word pairs.
    Reduce dimensions with SVD and check the distance metrics on the word pairs again.
    """
    pairs = [
        ("women", "men"),
        ("women", "dogs"),
        ("men", "dogs"),
        ("feed", "like"),
        ("feed", "bite"),
        ("like", "bite"),
    ]

    with open('dist_sim_data.txt', 'r') as file:
        corpus = file.read().splitlines()

     # Extract unique words
    words = list(set(" ".join(corpus).split()))
    word_to_id = {word: i for i, word in enumerate(words)}

    # Initialize co-occurrence matrix
    C = np.zeros((len(words), len(words)), dtype=int)

    # Fill co-occurrence matrix
    for sentence in corpus:
        tokens = sentence.split()
        for i, token in enumerate(tokens):
            if token in words:
                for j in range(max(0, i-1), min(i+2, len(tokens))):
                    if tokens[j] in words and i != j:
                        C[word_to_id[token], word_to_id[tokens[j]]] += 1

    # Smooth and scale the co-occurrence matrix
    C = (C + 1) * 10
    print("Co-occurrence matrix:\n", C)

    # Compute PPMI matrix
    row_sums = C.sum(axis=1, keepdims=True)
    col_sums = C.sum(axis=0, keepdims=True)
    total = float(C.sum())
    PPMI = np.maximum(np.log((C * total) / (row_sums @ col_sums)), 0)
    print("PPMI matrix:\n", PPMI)

    # Evaluate word similarity before dimension reduction
    print("Euclidean distances before SVD:")
    for word1, word2 in pairs:
        vec1 = C[word_to_id[word1]]
        vec2 = C[word_to_id[word2]]
        print(f"{word1} - {word2}: {euclidean_distance(vec1, vec2)}")

    # SVD
    U, E, Vt = scipy_linalg.svd(PPMI, full_matrices=False)
    E = np.diag(E)
    assert np.allclose(PPMI, U.dot(E).dot(Vt)), "SVD reconstruction failed"

    # Reduce dimensions
    V = Vt.T
    reduced_PPMI = PPMI.dot(V[:, :3])

    # Evaluate word similarity after dimensionality reduction
    print()
    print("Euclidean distances after SVD:")
    for word1, word2 in pairs:
        vec1 = reduced_PPMI[word_to_id[word1]]
        vec2 = reduced_PPMI[word_to_id[word2]]
        print(f"{word1} - {word2}: {euclidean_distance(vec1, vec2)}")


#End of part 1

def load_vectors(filename):
    vectors = {}
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vec = np.array([float(val) for val in parts[1:]])
            vectors[word] = vec
    return vectors

# Generate synonym test questions
def generate_synonym_questions(vectors, num_choices=5, filename='synonym_questions.json'):
    if os.path.exists(filename):
        # Load questions if they have already been generated and saved
        with open(filename, 'r', encoding='utf-8') as f:
            questions = json.load(f)
    else:
        # Continue with question generation as before
        with open('EN_syn_verb.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        verbs = {}
        for line in lines:
            input_word, answer_suggestion = line.strip().split('\t')
            normalized_input_word = input_word.replace("to_", "").replace("_", " ")
            normalized_answer_suggestion = answer_suggestion.replace("to_", "").replace("_", " ")
            verbs.setdefault(normalized_input_word, set()).add(normalized_answer_suggestion)
        
        questions = []
        all_words = set(vectors.keys())  # Use all words in vectors for potential wrong answers

        for verb, synonyms in verbs.items():
            synonyms = set(filter(lambda x: x in vectors, synonyms))  # Ensure synonyms are in vectors
            if len(synonyms) < 1:
                continue
            
            correct = random.choice(list(synonyms))
            wrong_candidates = list(all_words - synonyms - {verb})
            if len(wrong_candidates) < num_choices - 1:
                continue
            
            wrong = random.sample(wrong_candidates, num_choices - 1)
            options = wrong + [correct]
            #question = {'verb': verb, 'options': options, 'correct': correct}
            questions.append((verb, options, correct))

            if len(questions) == 1000:  # Limit to 1,000 questions
                break

        # Save the generated questions to a file
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(questions, f, indent=4)
            
    return questions

def synonym_test(questions, vectors):
    correct_euclidean = 0
    correct_cosine = 0

    for verb, choices, correct in questions:
        if verb not in vectors or any(choice not in vectors for choice in choices):
            continue  # Skip if any word is missing
        
        verb_vec = vectors[verb]
        distances = []
        similarities = []
        
        for choice in choices:
            choice_vec = vectors[choice]
            distances.append(euclidean_distance(verb_vec, choice_vec))
            similarities.append(cosine_similarity(verb_vec, choice_vec))
        
        # Identify the best choices
        best_euclidean = choices[np.argmin(distances)]
        best_cosine = choices[np.argmax(similarities)]
        
        if best_euclidean == correct:
            correct_euclidean += 1
        if best_cosine == correct:
            correct_cosine += 1
    
    total_questions = len(questions)
    return (correct_euclidean / total_questions, correct_cosine / total_questions)


def run_synonym_test():
    """
    Sets up the synonym test, loads the word embeddings and runs the evaluation.
    Prints the overall accuracy of the synonym task.
    """
    # Load vectors
    classic_vectors = load_vectors('EN-wform.w.2.ppmi.svd.500-filtered.txt')
    word2vec_vectors = load_vectors('GoogleNews-vectors-negative300-filtered.txt')

    # Generate questions
    classic_questions = generate_synonym_questions(classic_vectors)
    word2vec_questions = generate_synonym_questions(word2vec_vectors)

    # Run tests
    classic_accuracy = synonym_test(classic_questions, classic_vectors)
    word2vec_accuracy = synonym_test(word2vec_questions, word2vec_vectors)

    print("Synonym accuracy with Classic Vectors (Euclidian, cosine):", classic_accuracy)
    print("Synonym accuracy with Word2Vec Vectors (Euclidian, cosine):", word2vec_accuracy)

#START of Sat part of the homework
    
def parse_sat_questions():
    questions = []
    with open('SAT-package-V3.txt', 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#") and not line.startswith("190 FROM") and not line.startswith("KS type:") and line.strip() != '']
    
    question_blocks = []  # List to hold blocks of lines corresponding to each question
    current_block = []  # Current block of lines being processed
    
    # Group lines into blocks per question
    for line in lines:
        if line in 'abcde':  # This is the correct answer indicator
            current_block.append(line)  # Add the answer letter to the current block
            question_blocks.append(current_block)  # Append the complete block, including the answer, to question_blocks
            current_block = []  # Start a new block for the next question
        else:
            current_block.append(line)  # Add question lines to the current block

    # Check if there is an unfinished block left after the loop
    if current_block:
        question_blocks.append(current_block)
    
    # Parse each block into questions
    for block in question_blocks:
        
        stem_pair = tuple(block[0].split()[:2])  # First line of block is the stem pair
        choices = [tuple(block[i].split()[:2]) for i in range(1, 6)]  # Next five lines are choices
       
        correct_answer_letter = block[-1]  # The last element of block is the correct answer letter
        
        # Map the correct answer letter to its corresponding tuple in choices
        letter_to_index = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4}
        correct_answer_index = letter_to_index[correct_answer_letter.lower()]
        correct_answer_tuple = choices[correct_answer_index]
        
        questions.append((stem_pair, choices, correct_answer_letter, correct_answer_tuple))

    # for i, question in enumerate(questions[:10]):
    #     stem_pair, choices, correct_answer_letter, correct_answer_tuple = question
    #     print(f"Question {i+1}: Stem: {stem_pair}")
    #     for j, choice in enumerate(choices):
    #         print(f"  Choice {chr(97+j)}: {choice}")
    #     print(f"  Correct Answer Letter: {correct_answer_letter}")
    #     print(f"  Correct Answer Tuple: {correct_answer_tuple}")
    #     print("")

    return questions

def sat_test(questions, vectors):
    #HERE IS WHAT YOU NEED TO COMPLETE 
    correct_count = 0

    for question in questions:
        stem_pair, choices, correct_answer_letter, correct_answer_tuple = question

        # Skip the question if either word of the stem pair is not in vectors
        if stem_pair[0] not in vectors or stem_pair[1] not in vectors:
            continue

        stem_vector = vectors[stem_pair[1]] - vectors[stem_pair[0]]
        max_similarity = -np.inf
        chosen_answer = None

        for choice in choices:
            # Skip the choice if either word is not in vectors
            if choice[0] not in vectors or choice[1] not in vectors:
                continue

            choice_vector = vectors[choice[1]] - vectors[choice[0]]
            similarity = 1 - cosine(stem_vector, choice_vector)

            if similarity > max_similarity:
                max_similarity = similarity
                chosen_answer = choice

        if chosen_answer == correct_answer_tuple:
            correct_count += 1

    # Calculate accuracy
    accuracy = correct_count / len(questions) if questions else 0
    return accuracy


def run_sat_test():
    """
    Sets up the SAT test, loads the word embeddings and runs the evaluation.
    Prints the overall accuracy of the SAT task.
    """
    sat_questions = parse_sat_questions()
    classic_vectors = load_vectors('EN-wform.w.2.ppmi.svd.500-filtered.txt')
    word2vec_vectors = load_vectors('GoogleNews-vectors-negative300-filtered.txt')

    classic_accuracy = sat_test(sat_questions, classic_vectors)
    word2vec_accuracy = sat_test(sat_questions, word2vec_vectors)

    print("SAT accuracy with Classic Vectors:", classic_accuracy)
    print("SAT accuracy with Word2Vec Vectors:", word2vec_accuracy)

def part2():
    """
    Runs the two tasks for part two of Homework 4.
    """
    run_synonym_test()
    run_sat_test()



if __name__ == "__main__":
    # DO NOT MODIFY HERE
    part1()
    part2()