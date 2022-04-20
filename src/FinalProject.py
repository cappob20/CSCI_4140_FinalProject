from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizer_v2 import rmsprop as rmsprop_v2
import numpy as np
import sys
from os.path import exists


#LOAD TEXT
filename = "text.txt"
rawText = open(filename, 'r', encoding='utf-8').read()
rawText = rawText.lower()

#REMOVE NUMBERS
rawText = ''.join(c for c in rawText if not c.isdigit() if not c == '\ufeff' if not c == '\n')

#How many total characters do we have in our training text?
chars = sorted(list(set(rawText))) #List of every character
print(chars)

#Character sequences must be encoded as integers. 
#Each unique character will be assigned an integer value. 
#Create a dictionary of characters mapped to integer values
charToInt = dict((c, i) for i, c in enumerate(chars))

#Do the reverse so we can print our predictions in characters and not integers
intToChar = dict((i, c) for i, c in enumerate(chars))

# summarize the data
noChars = len(rawText)
noVocab = len(chars)


########################
#Now that we have characters we can create input/output sequences for training
#Remember that for LSTM input and output can be sequences... hence the term seq2seq


seqLength = 60  #Length of each input sequence
step = 5   #Instead of moving 1 letter at a time, try skipping a few. 
sentences = []    # X values (Sentences)
nextChars = []   # Y values. The character that follows the sentence defined as X
for i in range(0, noChars - seqLength, step):  #step=1 means each sentence is offset just by a single letter
    sentences.append(rawText[i: i + seqLength])  #Sequence in
    nextChars.append(rawText[i + seqLength])  #Sequence out
noPatterns = len(sentences)    


x = np.zeros((len(sentences), seqLength, noVocab), dtype=bool)
y = np.zeros((len(sentences), noVocab), dtype=bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, charToInt[char]] = 1
    y[i, charToInt[nextChars[i]]] = 1


#Basic model with LSTM
model = Sequential()
model.add(LSTM(128, input_shape=(seqLength, noVocab)))
model.add(Dense(noVocab, activation='softmax'))

optimizer = rmsprop_v2.RMSprop(learning_rate=0.00146)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


# checks if the user wants to retrain before continuing, if there does not exist a training already.

from keras.callbacks import ModelCheckpoint
check = 1
if exists("saved_weights.h5"):
    userInput = input("Do you want to retrain data? (Y/n): ")
    if (userInput.lower() == "y"):
        check = 1
    else:
        check = 0
if check == 1:
    filepath="saved_weights/saved_weights-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')

    callbacksList = [checkpoint]


    # Fit the model

    history = model.fit(x, y,
            batch_size=128,
            epochs=20,
            callbacks=callbacksList)

    model.save('saved_weights.h5')


#The prediction results is probabilities for each of the characters at a specific
#point in sequence. It picks the three highest (random if values are the same)
#probabilities

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



#Prediction
# load the network weights
model.load_weights("saved_weights.h5")

userInput = ""
currText = ""
while True:
    sys.stdout.write("\nSelect character from the list. If nothing is selected, the selection will be reset. Press Tab then Enter to leave.\n")
    prediction = np.zeros((1, seqLength, noVocab))
    for i, char in enumerate(currText):
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
    
