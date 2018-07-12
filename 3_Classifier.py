#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 10:27:39 2018

@author: aditya
"""

import pandas as pd
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.externals import joblib
import csv

number_of_files = 12

min_author_count = 100
max_author_count = 1000
gender = ["M","F"]
gender_phrase = ["I am a man","I am a woman"]

text_data = []

for j in range(len(gender)):
    start= datetime.now()

    filename1 = "../Data/" + gender[j] + "_2017_0"
    
    input_data = []
    for i in range(number_of_files):
        try:
            filename = filename1 + str(i+1) + ".csv"
            if i >=9:
                filename = filename1[:-1] + str(i+1) + ".csv"
    #        input_data.append(pd.read_csv(filename,encoding='utf-8',quoting=csv.QUOTE_NONE))
            input_data.append(pd.read_csv(open(filename,'rU'), encoding='utf-8',engine="c",dtype={'author':str,'body':str},low_memory=False))      
            print(filename)
        except:
            print("ERROR Loading File: " + filename)

    all_input_data = pd.concat(input_data,axis=0)
    
    allowed_authors = []
    for i in range(len(input_data)):
        allowed_authors += list(input_data[i][input_data[i].body.str.contains(gender_phrase[j])==True].author)
    
    allowed_authors = [str(x) for x in np.unique(allowed_authors)]
#    
    for i in range(len(input_data)):
        input_data[i] = input_data[i][input_data[i]["author"].isin(allowed_authors)]
    
    author_count_list = []
    for i in range(len(input_data)):
        author_count_list.append(input_data[i].groupby("author").count())
    
    author_count = author_count_list[0]
    for i in range(1,len(author_count_list)):
        author_count = author_count.add(author_count_list[i],fill_value=0)
    author_high_count = list(author_count[max_author_count < author_count["body"]].index)
    author_count = author_count[author_count["body"]>min_author_count]
    author_count = author_count[max_author_count > author_count["body"]]
    allowed_authors = list(author_count.index)
    
    for i in range(len(input_data)):
        input_data[i] = input_data[i][input_data[i]["author"].isin(allowed_authors)]
        input_data[i] = pd.DataFrame(input_data[i].groupby('author')['body'].apply(lambda x: "{%s}" % '\n'.join(x.astype(str))))

    extra_input = []
    for author in author_high_count:
        author_df = all_input_data[all_input_data["author"].isin([author])]
        extra_input.append(pd.DataFrame(author_df.sample(n=max_author_count)))
    extra_input = pd.concat(extra_input,axis=0)
    
#    final_data = pd.concat(input_data,axis=0)
    final_data = pd.concat(input_data + [extra_input],axis=0)
    final_data = pd.DataFrame(final_data.groupby('author')['body'].apply(lambda x: "{%s}" % '\n'.join(x.astype(str))))
    final_data = final_data.body.str.replace('I am a man', ' ')
    final_data = pd.DataFrame(final_data)
    final_data = list(final_data.fillna(' ')["body"])
    final_data = [a[2:-2] for a in final_data]
    text_data.append(final_data)
    end = datetime.now()
    
    print(end-start)

#%%

test_index_1 = int(len(text_data[0]) * 0.9)
test_index_2 = int(len(text_data[1]) * 0.9)

M_data_train = text_data[0][:test_index_1]
F_data_train = text_data[1][:test_index_2]
M_data_test = text_data[0][test_index_1:]
F_data_test = text_data[1][test_index_2:]
data = M_data_train + F_data_train

y_train = []
for element in M_data_train:
    y_train.append(0)
for element in F_data_train:
    y_train.append(1)

y_test = []
for element in M_data_test:
    y_test.append(0)
for element in F_data_test:
    y_test.append(1)

start = datetime.now()
cv = CountVectorizer(stop_words = "english",lowercase = True,ngram_range=(1,3),min_df=10,binary = True)
#cv = TfidfVectorizer(stop_words = "english",lowercase = True,ngram_range=(1,3))
X_data_train = cv.fit_transform(data)
end = datetime.now()
joblib.dump(cv, 'CV_Gender.pkl') 
print(end-start)
print("PREPROCESSING DONE - STARTING TRAINING")

#%%

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data_test = M_data_test + F_data_test
X_data_test = cv.transform(data_test).toarray()

clf = LogisticRegression()
clf.fit(X_data_train,y_train)
joblib.dump(clf, 'CLF_Gender.pkl')
y_pred = clf.predict(X_data_test)
print(accuracy_score(y_pred,y_test))
