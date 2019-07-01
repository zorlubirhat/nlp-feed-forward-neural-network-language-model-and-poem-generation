import json
import numpy as np
import dynet as dy
import math

# Path of reading .json file
dataset_name = 'unim_poem'
dataset_path = dataset_name + '.json'

poem_data = []
poem_data_tagging = []
count_poem_bigrams = {}
words_list = []

# Read json file
with open(dataset_path) as json_file:
    data = json.load(json_file)
    for p in data:
        poem_data.append(p['poem'])

# Adding START and END tags to the poems
for poem in poem_data:
    string = ""
    string += 'START '
    words_list.append('START')

    poem_sentences = poem.split("\n")
    for i in range(len(poem_sentences) - 1):
        one_sentence = poem_sentences[i].split(" ")
        for index in range(len(one_sentence) - 1):
            string += one_sentence[index] + " "
        string += one_sentence[len(one_sentence) - 1]

        for word in one_sentence:
            words_list.append(word)

        string += ' \n '
        words_list.append("\n")

    one_sentence = poem_sentences[len(poem_sentences) - 1].split(" ")
    for index in range(len(one_sentence) - 1):
        string += one_sentence[index] + " "
    string += one_sentence[len(one_sentence) - 1]

    for word in one_sentence:
        words_list.append(word)

    string += ' END'
    words_list.append('END')

    poem_data_tagging.append(string)

print("Tagging completed.")

# Creating bigrams from each sentence of each poem
poem_bigrams = [b for l in poem_data_tagging for b in zip(l.split(" ")[:-1], l.split(" ")[1:])]

# Counting bigrams pair
for poem_bigram in poem_bigrams:
    count_poem_bigrams[poem_bigram] = count_poem_bigrams.get(poem_bigram, 0) + 1

print("Bigrams created.")

# Initializing DyNet model
model = dy.Model()

# Creating unique words list and setting vocabulary size
unique_words_list = list(set(words_list))
vocab_size = len(unique_words_list)

# Indexing word to id and id to word
word_to_id = {x: i for i, x in enumerate(unique_words_list)}
id_to_word = {i: x for i, x in enumerate(unique_words_list)}

print("Indexing completed.")

words_index = []
for x, y in word_to_id.items():
    words_index.append(y)

# Creating one hot vector from each word
one_hot_vectors = []
for index in words_index:
    vector = dy.one_hot(d=vocab_size, idx=index).value()
    one_hot_vectors.append(vector)

print("One Hot Vectors created.")

# Creating data, indexes for model which enter input and output vector
data = []
indexes = []
for bigrams in poem_bigrams:
    index_word1 = word_to_id.get(bigrams[0])
    index_word2 = word_to_id.get(bigrams[1])
    vectors_of_words = [one_hot_vectors[index_word1], one_hot_vectors[index_word2]]
    indexes_of_words = [index_word1, index_word2]
    data.append(vectors_of_words)
    indexes.append(indexes_of_words)

print("Data and Indexes created.")

# Setting the input, hidden and output size
bigram_size = len(poem_bigrams)
input_size = vocab_size
hidden_size = 10
output_size = vocab_size

# Adding parameters to the model
W = model.add_parameters((hidden_size, input_size))
b = model.add_parameters(hidden_size)
U = model.add_parameters((output_size, hidden_size))
d = model.add_parameters(output_size)

# Initializing trainer with SimpleSGDTrainer
trainer = dy.SimpleSGDTrainer(model)

# Setting size of epoch
EPOCHS = 10

# Training the model accorgind to epoch size and one hot vectors of each bigram pairs
for epoch in range(EPOCHS):
    epoch_loss = 0.0
    for index in range(len(data)):
        dy.renew_cg()
        x = dy.inputVector(data[index][0])

        # prediction
        h = dy.tanh(W * x + b)
        yhat = U * h + d

        # calculate loss
        loss = dy.pickneglogsoftmax(yhat, indexes[index][1])

        epoch_loss += loss.scalar_value()

        loss.backward()
        trainer.update()

    print("Epoch %d. loss = %f" % (epoch, (epoch_loss / bigram_size)))

# Saving model
model.save(dataset_name + ".model")
print("Model saved.")

# # Loading model
# l_model = dy.ParameterCollection()
#
# l_W = l_model.add_parameters((hidden_size, input_size))
# l_b = l_model.add_parameters(hidden_size)
# l_U = l_model.add_parameters((output_size, hidden_size))
# l_d = l_model.add_parameters(output_size)
#
# l_model.populate(dataset_name + ".model")
# print("Model loaded.")


# Return output vector according to input vector
def return_output(o_h_vecs, g_word):
    input_vector = o_h_vecs[word_to_id[g_word]]
    dy.renew_cg()
    n_input = dy.inputVector(input_vector)

    n_h = dy.tanh(W * n_input + b)
    n_output = U * n_h + d

    return n_output.npvalue()


# Softmax for output probabilities
def normalize(arr):
    return list(np.exp(arr)/np.sum(np.exp(arr)))


# Creating poem according to given line size
# Returns created poem and probability of poem
def create_poem(n):
    total_probability = 0
    new_line_size = 0

    prev_word = "START"

    n_poem = []

    while new_line_size < int(n):

        n_output = return_output(one_hot_vectors, prev_word)

        probabilities = normalize(n_output)

        predicted = np.random.choice(unique_words_list, p=probabilities)

        word_probability = probabilities[word_to_id[predicted]]

        total_probability += math.log(word_probability, 2)

        if predicted == "\n":
            new_line_size += 1

        if predicted == "END":
            break

        n_poem.append(predicted)

        prev_word = predicted

    return n_poem, total_probability


# Printing the poem
def print_poem(g_poem):
    for g_word in g_poem:
        if g_word == "\n":
            print(g_word, end="")
        else:
            print(g_word, end=" ")


# Calculating the perplexity of poem
def calculate_perplexity(g_poem, p_probability):
    size = len(g_poem)
    log_sum = 0
    log_sum -= p_probability
    return math.pow(2, (float(log_sum) / size))


# Creating, printing 5 different poems and calculating the perplexity of each poem
for i in range(5):
    poem_length = input("Please enter line size of poem: ")
    poem, poem_probability = create_poem(poem_length)
    poem_perplexity = calculate_perplexity(poem, poem_probability)
    print("{}. Poem".format(i+1))
    print_poem(poem)
    print()
    print("{}. poem's perplexity is {}".format(i+1, poem_perplexity))
    print("*-*-*-*-*-*-*-*-*-*-*-*-*-*-*")
