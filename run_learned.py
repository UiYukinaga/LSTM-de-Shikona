from keras.models import Sequential,load_model
from keras.layers import Dense, Activation, LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import os

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probs = np.random.multinomial(1, preds, 1)
    return np.argmax(probs)



args = sys.argv
seed = args[1]

path = "./shikona.txt"
bindata = open(path, "rb").read()
text = bindata.decode("utf-8")
print("Size of text: ",len(text))
chars = sorted(list(set(text)))
print("Total chars :",len(chars))

#辞書を作成する
char_indices = dict((c,i) for i,c in enumerate(chars))
indices_char = dict((i,c) for i,c in enumerate(chars))

model = load_model('shikona.h5')
model.summary()

#3文字の次の1文字を学習させる. 2文字ずつずらして3文字と1文字というセットを作る
maxlen = 2

num = 1
print("-----次で始まる四股名を生成: " + seed)
for diversity in [0.2, 1.0, 1.2, 1.5]:
    index = maxlen * (-1)
    sentence = seed[index:]
    generated = seed
    
    #次の文字を予測して足す
    for i in range(99):
        x = np.zeros((1,maxlen,len(chars)))
        for t,char in enumerate(sentence):
            x[0, t, char_indices[char]] = 1

        preds = model.predict(x, verbose =9)[0] #次の文字を予測
        next_index = sample(preds, diversity)
        next_char = indices_char[next_index]

        # <マジックハンドアシスト>
        # seedの次の文字が見つからなかったとき、"ノ"の字で補完する
        # ----------------------------------------------------
        if '\n' in next_char:
            if len(generated) <= len(seed):
                next_char = 'ノ'
        # ----------------------------------------------------
        
        if '\n' in next_char:
            print('その' + str(num) + '> ' + generated)
            num += 1
            break
        else:
            generated += next_char
            sentence = sentence[1:] + next_char
    print()