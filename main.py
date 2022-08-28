import eel
import re
import pandas as pd
import numpy as np
import numpy as np
import re

from tensorflow.keras.layers import Dense, LSTM, Input, Dropout, Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences

eel.init("web")



@eel.expose
def take_py(txt_in):
    global txt
    txt = txt_in
    print(txt)


#take_py()

Щавайка-аэроплан, [28.08.2022 12:18]
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


with open('1.txt', 'r', encoding='utf-8') as f:
    texts_1 = f.read()
    texts_1 = texts_1.replace('\ufeff', '')  # убираем первый невидимый символ
    texts_1 = re.sub(r'[^А-я0-9 \n]', '', texts_1)  # заменяем все символы кроме кириллицы на пустые символы

with open('2.txt', 'r', encoding='utf-8') as f:
    texts_2 = f.read()
    texts_2 = texts_2.replace('\ufeff', '')  # убираем первый невидимый символ
    texts_2 = re.sub(r'[^А-я0-9 \n]', '', texts_2)  # заменяем все символы кроме кириллицы на пустые символы

with open('3.txt', 'r', encoding='utf-8') as f:
    texts_3 = f.read()
    texts_3 = texts_3.replace('\ufeff', '')  # убираем первый невидимый символ
    texts_3 = re.sub(r'[^А-я0-9 \n]', '', texts_3)  # заменяем все символы кроме кириллицы на пустые символы

with open('4.txt', 'r', encoding='utf-8') as f:
    texts_4 = f.read()
    texts_4 = texts_4.replace('\ufeff', '')  # убираем первый невидимый символ
    texts_4 = re.sub(r'[^А-я0-9 \n]', '', texts_4)  # заменяем все символы кроме кириллицы на пустые символы

with open('5.txt', 'r', encoding='utf-8') as f:
    texts_5 = f.read()
    texts_5 = texts_5.replace('\ufeff', '')  # убираем первый невидимый символ
    texts_5 = re.sub(r'[^А-я0-9 \n]', '', texts_5)  # заменяем все символы кроме кириллицы на пустые символы

with open('6.txt', 'r', encoding='utf-8') as f:
    texts_6 = f.read()
    texts_6 = texts_6.replace('\ufeff', '')  # убираем первый невидимый символ
    texts_6 = re.sub(r'[^А-я0-9 \n]', '', texts_6)  # заменяем все символы кроме кириллицы на пустые символы

with open('7.txt', 'r', encoding='utf-8') as f:
    texts_7 = f.read()
    texts_7 = texts_7.replace('\ufeff', '')  # убираем первый невидимый символ
    texts_7 = re.sub(r'[^А-я0-9 \n]', '', texts_7)  # заменяем все символы кроме кириллицы на пустые символы

with open('8.txt', 'r', encoding='utf-8') as f:
    texts_8 = f.read()
    texts_8 = texts_8.replace('\ufeff', '')  # убираем первый невидимый символ
    texts_8 = re.sub(r'[^А-я0-9 \n]', '', texts_8)  # заменяем все символы кроме кириллицы на пустые символы

with open('9.txt', 'r', encoding='utf-8') as f:
    texts_9 = f.read()
    texts_9 = texts_9.replace('\ufeff', '')  # убираем первый невидимый символ
    texts_9 = re.sub(r'[^А-я0-9 \n]', '', texts_9)  # заменяем все символы кроме кириллицы на пустые символы

with open('10.txt', 'r', encoding='utf-8') as f:
    texts_10 = f.read()
    texts_10 = texts_10.replace('\ufeff', '')  # убираем первый невидимый символ
    texts_10 = re.sub(r'[^А-я0-9 \n]', '', texts_10)  # заменяем все символы кроме кириллицы на пустые символы

with open('11.txt', 'r', encoding='utf-8') as f:
    texts_11 = f.read()
    texts_11 = texts_11.replace('\ufeff', '')  # убираем первый невидимый символ
    texts_11 = re.sub(r'[^А-я0-9 \n]', '', texts_11)  # заменяем все символы кроме кириллицы на пустые символы

with open('12.txt', 'r', encoding='utf-8') as f:
    texts_12 = f.read()
    texts_12 = texts_12.replace('\ufeff', '')  # убираем первый невидимый символ
    texts_12 = re.sub(r'[^А-я0-9 \n]', '', texts_12)  # заменяем все символы кроме кириллицы на пустые символы

with open('13.txt', 'r', encoding='utf-8') as f:
    texts_13 = f.read()
    texts_13 = texts_13.replace('\ufeff', '')  # убираем первый невидимый символ
    texts_13 = re.sub(r'[^А-я0-9 \n]', '', texts_13)  # заменяем все символы кроме кириллицы на пустые символы

with open('14.txt', 'r', encoding='utf-8') as f:
    texts_14 = f.read()
    texts_14 = texts_14.replace('\ufeff', '')  # убираем первый невидимый символ
    texts_14 = re.sub(r'[^А-я0-9 \n]', '', texts_14)  # заменяем все символы кроме кириллицы на пустые символы

Щавайка-аэроплан, [28.08.2022 12:18]
with open('15.txt', 'r', encoding='utf-8') as f:
    texts_15 = f.read()
    texts_15 = texts_15.replace('\ufeff', '')  # убираем первый невидимый символ
    texts_15 = re.sub(r'[^А-я0-9 \n]', '', texts_15)  # заменяем все символы кроме кириллицы на пустые символы

with open('16.txt', 'r', encoding='utf-8') as f:
    texts_16 = f.read()
    texts_16 = texts_16.replace('\ufeff', '')  # убираем первый невидимый символ
    texts_16 = re.sub(r'[^А-я0-9 \n]', '', texts_16)  # заменяем все символы кроме кириллицы на пустые символы

with open('17.txt', 'r', encoding='utf-8') as f:
    texts_17 = f.read()
    texts_17 = texts_17.replace('\ufeff', '')  # убираем первый невидимый символ
    texts_17 = re.sub(r'[^А-я0-9 \n]', '', texts_17)  # заменяем все символы кроме кириллицы на пустые символы

with open('18.txt', 'r', encoding='utf-8') as f:
    texts_18 = f.read()
    texts_18 = texts_18.replace('\ufeff', '')  # убираем первый невидимый символ
    texts_18 = re.sub(r'[^А-я0-9 \n]', '', texts_18)  # заменяем все символы кроме кириллицы на пустые символы

with open('19.txt', 'r', encoding='utf-8') as f:
    texts_19 = f.read()
    texts_19 = texts_19.replace('\ufeff', '')  # убираем первый невидимый символ
    texts_19 = re.sub(r'[^А-я0-9 \n]', '', texts_19)  # заменяем все символы кроме кириллицы на пустые символы

with open('20.txt', 'r', encoding='utf-8') as f:
    texts_20 = f.read()
    texts_20 = texts_20.replace('\ufeff', '')  # убираем первый невидимый символ
    texts_20 = re.sub(r'[^А-я0-9 \n]', '', texts_20)  # заменяем все символы кроме кириллицы на пустые символы

with open('21.txt', 'r', encoding='utf-8') as f:
    texts_21 = f.read()
    texts_21 = texts_21.replace('\ufeff', '')  # убираем первый невидимый символ
    texts_21 = re.sub(r'[^А-я0-9 \n]', '', texts_21)  # заменяем все символы кроме кириллицы на пустые символы

with open('22.txt', 'r', encoding='utf-8') as f:
    texts_22 = f.read()
    texts_22 = texts_22.replace('\ufeff', '')  # убираем первый невидимый символ
    texts_22 = re.sub(r'[^А-я0-9 \n]', '', texts_22)  # заменяем все символы кроме кириллицы на пустые символы

with open('23.txt', 'r', encoding='utf-8') as f:
    texts_23 = f.read()
    texts_23 = texts_23.replace('\ufeff', '')  # убираем первый невидимый символ
    texts_23 = re.sub(r'[^А-я0-9 \n]', '', texts_23)  # заменяем все символы кроме кириллицы на пустые символы

with open('24.txt', 'r', encoding='utf-8') as f:
    texts_24 = f.read()
    texts_24 = texts_24.replace('\ufeff', '')  # убираем первый невидимый символ
    texts_24 = re.sub(r'[^А-я0-9 \n]', '', texts_24)  # заменяем все символы кроме кириллицы на пустые символы

with open('25.txt', 'r', encoding='utf-8') as f:
    texts_25 = f.read()
    texts_25 = texts_25.replace('\ufeff', '')  # убираем первый невидимый символ
    texts_25 = re.sub(r'[^А-я0-9 \n]', '', texts_25)  # заменяем все символы кроме кириллицы на пустые символы

with open('26.txt', 'r', encoding='utf-8') as f:
    texts_26 = f.read()
    texts_26 = texts_26.replace('\ufeff', '')  # убираем первый невидимый символ
    texts_26 = re.sub(r'[^А-я0-9 \n]', '', texts_26)  # заменяем все символы кроме кириллицы на пустые символы

with open('27.txt', 'r', encoding='utf-8') as f:
    texts_27 = f.read()
    texts_27 = texts_27.replace('\ufeff', '')  # убираем первый невидимый символ
    texts_27 = re.sub(r'[^А-я0-9 \n]', '', texts_27)  # заменяем все символы кроме кириллицы на пустые символы

with open('28.txt', 'r', encoding='utf-8') as f:
    texts_28 = f.read()
    texts_28 = texts_28.replace('\ufeff', '')  # убираем первый невидимый символ
    texts_28 = re.sub(r'[^А-я0-9 \n]', '', texts_28)  # заменяем все символы кроме кириллицы на пустые символы

with open('29.txt', 'r', encoding='utf-8') as f:
    texts_29 = f.read()
    texts_29 = texts_29.replace('\ufeff', '')  # убираем первый невидимый символ
    texts_29 = re.sub(r'[^А-я0-9 \n]', '', texts_29)  # заменяем все символы кроме кириллицы на пустые символы

Щавайка-аэроплан, [28.08.2022 12:18]
with open('30.txt', 'r', encoding='utf-8') as f:
    texts_30 = f.read()
    texts_30 = texts_30.replace('\ufeff', '')  # убираем первый невидимый символ
    texts_30 = re.sub(r'[^А-я0-9 \n]', '', texts_30)  # заменяем все символы кроме кириллицы на пустые символы

with open('31.txt', 'r', encoding='utf-8') as f:
    texts_31 = f.read()
    texts_31 = texts_31.replace('\ufeff', '')  # убираем первый невидимый символ
    texts_31 = re.sub(r'[^А-я0-9 \n]', '', texts_31)  # заменяем все символы кроме кириллицы на пустые символы

with open('32.txt', 'r', encoding='utf-8') as f:
    texts_32 = f.read()
    texts_32 = texts_32.replace('\ufeff', '')  # убираем первый невидимый символ
    texts_32 = re.sub(r'[^А-я0-9 \n]', '', texts_32)  # заменяем все символы кроме кириллицы на пустые символы

with open('33.txt', 'r', encoding='utf-8') as f:
    texts_33 = f.read()
    texts_33 = texts_33.replace('\ufeff', '')  # убираем первый невидимый символ
    texts_33 = re.sub(r'[^А-я0-9 \n]', '', texts_33)  # заменяем все символы кроме кириллицы на пустые символы

with open('34.txt', 'r', encoding='utf-8') as f:
    texts_34 = f.read()
    texts_34 = texts_34.replace('\ufeff', '')  # убираем первый невидимый символ
    texts_34 = re.sub(r'[^А-я0-9 \n]', '', texts_34)  # заменяем все символы кроме кириллицы на пустые символы

with open('35.txt', 'r', encoding='utf-8') as f:
    texts_35 = f.read()
    texts_35 = texts_35.replace('\ufeff', '')  # убираем первый невидимый символ
    texts_35 = re.sub(r'[^А-я0-9 \n]', '', texts_35)  # заменяем все символы кроме кириллицы на пустые символы

with open('36.txt', 'r', encoding='utf-8') as f:
    texts_36 = f.read()
    texts_36 = texts_36.replace('\ufeff', '')  # убираем первый невидимый символ
    texts_36 = re.sub(r'[^А-я0-9 \n]', '', texts_36)  # заменяем все символы кроме кириллицы на пустые символы

with open('37.txt', 'r', encoding='utf-8') as f:
    texts_37 = f.read()
    texts_37 = texts_37.replace('\ufeff', '')  # убираем первый невидимый символ
    texts_37 = re.sub(r'[^А-я0-9 \n]', '', texts_37)  # заменяем все символы кроме кириллицы на пустые символы

with open('38.txt', 'r', encoding='utf-8') as f:
    texts_38 = f.read()
    texts_38 = texts_38.replace('\ufeff', '')  # убираем первый невидимый символ
    texts_38 = re.sub(r'[^А-я0-9 \n]', '', texts_38)  # заменяем все символы кроме кириллицы на пустые символы

with open('39.txt', 'r', encoding='utf-8') as f:
    texts_39 = f.read()
    texts_39 = texts_39.replace('\ufeff', '')  # убираем первый невидимый символ
    texts_39 = re.sub(r'[^А-я0-9 \n]', '', texts_39)  # заменяем все символы кроме кириллицы на пустые символы

texts = texts_1 + texts_2 + texts_3 + texts_4 + texts_5 + texts_6 + texts_7 + texts_8 + texts_9 + texts_10 + texts_11 + texts_12 + texts_13 + texts_14 + texts_15 + texts_16 + texts_17 + texts_18 + texts_19 + texts_20 + texts_21 + texts_22 + texts_23 + texts_24 + texts_25 + texts_26 + texts_27 + texts_28 + texts_29 + texts_30 + texts_31 + texts_32 + texts_33 + texts_34 + texts_35 + texts_36 + texts_37 + texts_38 + texts_39
count_1 = len(texts_1)
count_2 = len(texts_2)
count_3 = len(texts_3)
count_4 = len(texts_4)
count_5 = len(texts_5)
count_6 = len(texts_6)
count_7 = len(texts_7)
count_8 = len(texts_8)
count_9 = len(texts_9)
count_10 = len(texts_10)
count_11 = len(texts_11)
count_12 = len(texts_12)
count_13 = len(texts_13)
count_14 = len(texts_14)
count_15 = len(texts_15)
count_16 = len(texts_16)
count_17 = len(texts_17)
count_18 = len(texts_18)
count_19 = len(texts_19)
count_20 = len(texts_20)
count_21 = len(texts_21)
count_22 = len(texts_22)
count_23 = len(texts_23)
count_24 = len(texts_24)
count_25 = len(texts_25)
count_26 = len(texts_26)
count_27 = len(texts_27)
count_28 = len(texts_28)
count_29 = len(texts_29)
count_30 = len(texts_30)
count_31 = len(texts_31)
count_32 = len(texts_32)
count_33 = len(texts_33)
count_34 = len(texts_34)
count_35 = len(texts_35)
count_36 = len(texts_36)
count_37 = len(texts_37)
count_38 = len(texts_38)
count_39 = len(texts_39)

Щавайка-аэроплан, [28.08.2022 12:18]
total_lines = count_1 + count_2 + count_3 + count_4 + count_5 + count_6 + count_7 + count_8 + count_9 + count_10 + count_11 + count_12 + count_13 + count_14 + count_15 + count_16 + count_17 + count_18 + count_19 + count_20 + count_21 + count_22 + count_23 + count_24 + count_25 + count_26 + count_27 + count_28 + count_29 + count_30 + count_31 + count_32 + count_33 + count_34 + count_35 + count_36 + count_37 + count_38 + count_39

maxWordsCount = 2000000
tokenizer = Tokenizer(num_words=maxWordsCount, lower=True, split=' ', char_level=False)
tokenizer.fit_on_texts([texts])

dist = list(tokenizer.word_counts.items())
print(dist)

max_text_len = 2000000
data = tokenizer.texts_to_sequences([texts])
data_pad = pad_sequences(data, maxlen=max_text_len)
# res = to_categorical(data[0], num_classes=maxWordsCount)
# print(res.shape)
# res = np.array(data[0])

X = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0]] * count_1 + [
                 [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0]] * count_2 + [
                 [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0]] * count_3 + [
                 [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0]] * count_4 + [
                 [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0]] * count_5 + [
                 [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0]] * count_6 + [
                 [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0]] * count_7 + [
                 [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0]] * count_8 + [
                 [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0]] * count_9 + [
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0]] * count_10 + [
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0]] * count_12 + [
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0]] * count_13 + [
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0]] * count_14 + [
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0]] * count_1 + [
                 [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0]] * count_15 + [
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0]] * count_16 + [
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0]] * count_17 + [
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0]] * count_18 + [
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0]] * count_19 + [

Щавайка-аэроплан, [28.08.2022 12:18]
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0]] * count_20 + [
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0]] * count_21 + [
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0]] * count_22 + [
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0]] * count_23 + [
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0]] * count_24 + [
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0]] * count_25 + [
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0]] * count_26 + [
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0]] * count_27 + [
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0]] * count_28 + [
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0]] * count_29 + [
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                  0, 0, 0, 0, 0]] * count_30 + [
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                  0, 0, 0, 0, 0]] * count_31 + [
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                  0, 0, 0, 0, 0]] * count_32 + [
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                  0, 0, 0, 0, 0]] * count_33 + [
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                  0, 0, 0, 0, 0]] * count_34 + [
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  1, 0, 0, 0, 0]] * count_35 + [
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 1, 0, 0, 0]] * count_36 + [
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 1, 0, 0]] * count_37 + [
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 1, 0]] * count_38 + [
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 1]] * count_39)
Y = data_pad

model = Sequential()
model.add(Embedding(maxWordsCount, 256, input_length=max_text_len))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(64))
model.add(Dense(39, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam(0.0001))

history = model.fit(X, Y, batch_size=128, epochs=100)








#html файл
eel.start("main.html", mode="chrome", size=(1000, 850))