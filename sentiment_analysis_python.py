import pandas as pd
import numpy as np 
import re


!pip install stop-words

import stop_words

import nltk

from nltk.stem import  SnowballStemmer

stemmer_ru = SnowballStemmer('russian')
stemmer_eng = SnowballStemmer('english')

from nltk import word_tokenize

from nltk.corpus import stopwords
stop_ru = set(stopwords.words('russian'))
stop =  stop_ru | set(stop_words.get_stop_words('ru')) -set('год')

from sklearn.linear_model import LogisticRegression

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression 

from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold, GridSearchCV

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, recall_score, precision_score

from scipy.sparse import csr_matrix,hstack

df1.head(5)

df1 = pd.read_csv('data_universities.csv', engine='python', sep=';', names=['text', 'class'])

#df2 = pd.read_csv('clear_texts_01.csv', sep = ';', encoding = 'cp1251')

#df = pd.concat([df1, df2])
df = df1.copy()
df.drop_duplicates(inplace = True)
df['text'] = df['text'].str.replace('<br>', ' ')

# Разметим датафрейм 
# 1,2 - 0
# 3,4,5 - 1
df['sentiment'] = (df['class'] > 3).astype(int)

# drop class
df.drop('class', axis=1, inplace=True)

df.head(5)

df1.to_csv('classiffication_universities.csv')

df.sentiment.replace(2, 1, inplace = True)

df.sentiment.value_counts()

df

df['text'] = df['text'].str.replace(r'\[[a-zA-Zа-яА-Я\. \/\:\_\-0-9\|\>\<\?\+\,]+\] shared a link.', '')
df['text'] = df['text'].str.replace(r'\[[a-zA-Zа-яА-Я\. \/\:\_\-0-9\|\>\<\?\+\,]+\]', '')
df['text'] = df['text'].str.replace(r'(?:(?:https?|ftp):\/\/)?[\w/\-?=%.&]+\.[\w/\-?=%.&]+', '')
df['text'] = df['text'].str.replace('shared a ', '')
df['text'] = df['text'].str.replace(r'\W', ' ')

def string_transform(string):
    string = re.sub('[\W0-9]',' ',string)
    string = string.split()
    string = [stemmer_eng.stem(stemmer_ru.stem(i)) for i in string if i not in stop]

    return ' '.join(string)

df['new_text'] = df.text.astype(str).apply(string_transform)

words =(' '.join(df['text'])).split()
all_words = nltk.FreqDist(w.lower()for w in words)
word_features = [w for (w,ct) in all_words.most_common(20)]

word_features

df.head()

data = df['new_text']
tf_idf = TfidfVectorizer(max_features =10000, min_df=5, ngram_range = (1,2))

X_train, X_test, y_train, y_test  = train_test_split(data, df['sentiment'], test_size = 0.20, random_state = 42)
#model.fit(X_train, y_train, plot = False, eval_set = (X_test, y_test))

train_corpus = tf_idf.fit_transform(X_train)
test_corpus = tf_idf.transform(X_test)


#model = CatBoostClassifier(verbose = False, max_depth = 3, learning_rate=0.3, loss_function = 'MultiClass', iterations = 100, class_weights = [1, 10.64,4.84])
model1 = LogisticRegression(solver = 'sag', max_iter = 1000)

model1.fit(train_corpus, y_train)
prediction1 = model1.predict(test_corpus)
print('acc = {}, recall = {}'.format(accuracy_score(y_test, prediction1), recall_score(y_test, prediction1)))

confusion_matrix(y_test, prediction1)

from sklearn.svm import LinearSVC

model2 = LinearSVC(loss = 'squared_hinge',
    class_weight = {1:7},
    C = 0.2)

model2.fit(train_corpus, y_train)
prediction2 = model2.predict(test_corpus)
print('acc = {}, recall = {}'.
      format(accuracy_score(y_test, prediction2), recall_score(y_test, prediction2)))

confusion_matrix(y_test, prediction2)

y_test.value_counts()
