from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM
from keras.optimizer_v2 import rmsprop as rmsprop_v2
from keras.callbacks import ModelCheckpoint
import numpy as np
import sys
import random
from os.path import exists
import os

chars = None
charToInt = None
intToChar = None
noChars = None
noVocab = None

SEQ_LENGTH = 60  #Length of each input sequence

#character sampling utility method
def sample(preds):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds)
    expPreds = np.exp(preds) #exp of log (x), isn't this same as x??
    preds = expPreds / np.sum(expPreds)
    probas = np.random.multinomial(1, preds, 1)
    temp = np.ndarray.tolist(probas)

    num1 = np.argmax(np.random.multinomial(1, preds, 1))
    temp[0][num1] = 0
    num2 = np.argmax(np.random.multinomial(1, preds, 1))
    while num2 == num1:
        num2 = np.argmax(np.random.multinomial(1, preds, 1))
        temp[0][num2] = 0
    num3 = np.argmax(np.random.multinomial(1, preds, 1))
    while num3 == num2 or num3 == num1:
        num3 = np.argmax(np.random.multinomial(1, preds, 1))
        temp[0][num3] = 0
    nums = [num1, num2, num3]
    return nums

def load_train_text():
    global chars, charToInt, intToChar, noChars, noVocab

    #LOAD TEXT
    filename = "src/text.txt"
    rawText = open(filename, 'r', encoding='utf-8').read()
    rawText = rawText.lower()

    #How many total characters do we have in our training text?
    chars = sorted(list(set(rawText))) #List of every character
    #use only ascii character
    # chars = "abcdefghijklmnopqrstuvwxyz"
    print(chars)

    #REMOVE NUMBERS
    rawText = ''.join(c for c in rawText if c in chars)

    #Character sequences must be encoded as integers.
    #Each unique character will be assigned an integer value.
    #Create a dictionary of characters mapped to integer values
    charToInt = dict((c, i) for i, c in enumerate(chars))

    #Do the reverse so we can print our predictions in characters and not integers
    intToChar = dict((i, c) for i, c in enumerate(chars))

    # summarize the data
    noChars = len(rawText)
    noVocab = len(chars)

    return rawText

def load_test_data(fname):
    data = []
    with open(fname) as f:
        for line in f:
            inp = line[:-1]  # the last character is a newline
            inp = [c for c in inp if c in chars]
            data.append(inp)
    return data

def train(work_dir):
    rawText = load_train_text()

    ########################
    #Now that we have characters we can create input/output sequences for training
    #Remember that for LSTM input and output can be sequences... hence the term seq2seq

    step = 5   #Instead of moving 1 letter at a time, try skipping a few.
    sentences = []   # X values (Sentences)
    nextChars = []   # Y values. The character that follows the sentence defined as X
    for i in range(0, noChars - SEQ_LENGTH, step):  #step=1 means each sentence is offset just by a single letter
        sentences.append(rawText[i: i + SEQ_LENGTH])  #Sequence in
        nextChars.append(rawText[i + SEQ_LENGTH])  #Sequence out
    noPatterns = len(sentences)

    x = np.zeros((len(sentences), SEQ_LENGTH, noVocab), dtype=bool)
    y = np.zeros((len(sentences), noVocab), dtype=bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            x[i, t, charToInt[char]] = 1
        y[i, charToInt[nextChars[i]]] = 1

    #Basic model with LSTM
    model = Sequential()
    model.add(LSTM(128, input_shape=(SEQ_LENGTH, noVocab)))
    model.add(Dense(noVocab, activation='softmax'))

    optimizer = rmsprop_v2.RMSprop(learning_rate=0.00146)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    # checks if the user wants to retrain before continuing, if there does not exist a training already.
    check = 1
    if exists(os.path.join(work_dir, "saved_weights.h5")):
        userInput = input("Do you want to retrain data? (Y/n): ")
        if (userInput.lower() == "y"):
            check = 1
        else:
            check = 0
    if check == 1:
        filepath=os.path.join(work_dir, "saved_weights-{epoch:02d}-{loss:.4f}.hdf5")
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')

        callbacksList = [checkpoint]

        # Fit the model
        history = model.fit(x, y,
            batch_size=128,
            epochs=20,
            callbacks=callbacksList)

        model.save(os.path.join(work_dir, "saved_weights.h5"))


def test(model_path, input_file, output_file):
    #The prediction results is probabilities for each of the characters at a specific
    #point in sequence. It picks the three highest (random if values are the same)
    #probabilities

    #load training text so we can figure out our noVocab, etc
    load_train_text()

    #load test data (list of strings)
    input_data = load_test_data(input_file)

    #Prediction
    # load model
    model = load_model(model_path)

    preds = []
    for line in input_data:
        # this model just predicts a random character each time
        #one-hot encode input line
        test_x = np.zeros((1, SEQ_LENGTH, noVocab))
        for i, char in enumerate(line):
            #choose random character if input not in train set
            char = char if char in charToInt else random.choice(chars)
            test_x[0, i, charToInt[char]] = 1.
        #predict char probabilities
        pred_y = model.predict(test_x, verbose=0)[0]

        #pick top 3 characters, store in preds list
        top_3 = sample(pred_y)
        top_3 = [intToChar[x] for x in top_3]
        top_3 = ''.join(top_3)
        sys.stdout.write(top_3 + '\n')
        preds.append(top_3)

    #write predictions to output file
    # with open(output_file, 'wt') as f:
    with open(output_file, 'w', encoding="utf-8") as f:
        for p in preds:
            f.write('{}\n'.format(p))


def interactive(model_path):
    #The prediction results is probabilities for each of the characters at a specific
    #point in sequence. It picks the three highest (random if values are the same)
    #probabilities

    #load training text so we can figure out our noVocab, etc
    load_train_text()

    #Prediction
    # load the model
    model = load_model(model_path)

    userInput = ""
    currText = ""
    while True:
        sys.stdout.write("\nSelect character from the list. If nothing is selected, the selection will be reset. Press Tab then Enter to leave.\n")
        prediction = np.zeros((1, SEQ_LENGTH, noVocab))
        #remove unknown characters before prediction
        ct = [c for c in currText if c in chars]
        for i, char in enumerate(ct):
            prediction[0, i, charToInt[char]] = 1.
        preds = model.predict(prediction, verbose=0)[0]
        nums = sample(preds)
        top3 = "{}{}{}".format(intToChar[nums[0]], intToChar[nums[1]], intToChar[nums[2]])
        sys.stdout.write(top3 + '\n')
        userInput = input(currText).lower()
        currText += userInput
        if userInput == "":
            currText = ""
        if '\t' in userInput:
            break
