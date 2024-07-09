import numpy as np
import gensim
from gensim.scripts.glove2word2vec import glove2word2vec
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, TimeDistributed, Dense

# Step 1: Convert GloVe to Word2Vec format
glove_input_file = 'glove.6B/glove.6B.50d.txt'  # Replace with your GloVe file path
word2vec_output_file = 'glove.6B.50d.word2vec.txt'
glove2word2vec(glove_input_file, word2vec_output_file)

# Step 2: Load the converted model
word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)

# Training data
training_data = """john kills mary
john killed mary

who dies
who died

she is running
she was running

he runs
he ran

they are happy
they were happy

john kills
john killed

john kills whom
john killed whom

kill someone
killed someone

why are you crying
why were you crying

what are they doing
what were they doing

does john
did john

does john kill
did john kill

whom does john kill
whom did john kill

who kills
who killed

john cries
john cried

who kills mary
who killed mary

he dies
he died

they are running
they were running

they run
they ran

mary is crying
mary was crying

mary cries
mary cried

she is running
she was running

how are you
how were you

who am i
who was i

who are you
who were you

john runs fast
john ran fast

john is running slowly
john was running slowly

john runs slowly
john ran slowly

mary kills john
mary killed john

what are you doing
what were you doing

what is your name
what was your name

this is my book
this was my book

this one is mine
this one was mine

are you happy
were you happy

she never lies
she never lied

they never understand
they never understood

he understands
he understood

she understands me
she understood me

this is me
this was me

she is killed
she was killed

she never listens to me
she never listened to me

they laugh at me
they laughed at me

john throws a book at mary
john threw a book at mary

john throws a book
john threw a book

is it fine
was it fine

he laughs
he laughed

they laugh
they laughed

what are you here for
what were you here for

do you complete your homework
did you complete your homework

she has done her homework
she had done her homework

they have him quiet
they had him quiet

she is very kind
she was very kind

she does not know me
she did not know me

no-one completes their homework
no-one completed their homework

he do his homework
he did his homework

he can write his name
he could write his name

they are bothered
they were bothered

what is his name
what was his name

he goes to school
he went to school

she goes to office
she went to office

he is gone
he was gone

he leaves his school
he left his school

they are worried
they were worried

they are kind
they were kind

you are very good
you were very good

they are very nice
they were very nice

he walks very steadily
he walked very steadily

he walks
he walked

she is kind
she was kind

we are nice to him
we were nice to him

john does not know mary
john did not know mary

do you know me
did you know me"""

# Process the training data
length = 0
vocabulary = set(["_"])
for a in training_data.split("\n\n"):
    for b in a.split("\n"):
        length = max(length, len(b.split(" ")))
        for c in b.split(" "):
            vocabulary.add(c)
vocabulary = list(vocabulary)

def word_to_vec(word):
    if word in word2vec_model:
        return word2vec_model[word]
    else:
        return np.zeros(word2vec_model.vector_size)

x_train = []
for a in training_data.split("\n\n"):
    x_train.append([])
    for b in a.split("\n")[0].split(" "):
        x_train[-1].append(word_to_vec(b))
    while len(x_train[-1]) != length:
        x_train[-1].append(word_to_vec("_"))

y_train = []
for a in training_data.split("\n\n"):
    y_train.append([])
    for b in a.split("\n")[1].split(" "):
        y_train[-1].append(word_to_vec(b))
    while len(y_train[-1]) != length:
        y_train[-1].append(word_to_vec("_"))

x_new = x_train[-3:]
x_train = x_train[:-3]
y_train = y_train[:-3]

x_train = np.array(x_train)
y_train = np.array(y_train)
x_new = np.array(x_new)

# Determine input dimensions
input_dim = x_train.shape[-1]  # Dimension of the inner vectors
seq_length = x_train.shape[1]  # Length of each sequence

# Build BiLSTM model
model = Sequential([
    Bidirectional(LSTM(units=75, return_sequences=True), input_shape=(seq_length, input_dim)),
    TimeDistributed(Dense(units=input_dim))  # Apply Dense layer to each time step
])

model.compile(optimizer='adam', loss='mse')  # Mean squared error for regression

# Train the model
model.fit(x_train, y_train, epochs=50, batch_size=2)

# Predict the output
predictions = model.predict(x_new)

# Find the closest word in the vocabulary for each vector in the prediction
def closest_word(vector, model):
    try:
        return model.similar_by_vector(vector, topn=1)[0][0]
    except KeyError:
        return "UNKNOWN"

print("Input Sequence:")
for i, seq in enumerate(x_new):
    print(f"Question {i + 1}:")
    question_words = []
    for word_vector in seq:
        word = closest_word(word_vector, word2vec_model)
        if word != "UNKNOWN":
            question_words.append(word)
    print(" ".join(question_words))
    print("\nPredicted Completion:")
    predicted_words = []
    for pred_vector in predictions[i]:
        word = closest_word(pred_vector, word2vec_model)
        if word != "UNKNOWN":
            predicted_words.append(word)
    print(" ".join(predicted_words))
    print()

