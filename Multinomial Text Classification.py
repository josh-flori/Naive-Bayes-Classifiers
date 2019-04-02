""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                                       NOTES                                        

CountVectorizer builds a dictionary of features and transforms documents to feature vectors. 
Once fitted, the vectorizer has built a dictionary of feature indices:  The index value of a word 
in the vocabulary is linked to its frequency in the whole training corpus.

Text preprocessing, tokenizing and filtering of stopwords are all included in CountVectorizer, 
which builds a dictionary of features and transforms documents to feature vectors.

But word counts are not enough. We need Tf-idf.

The goal of using tf-idf instead of the raw frequencies of occurrence of a token in a given document 
is to scale down the impact of tokens that occur very frequently in a given corpus and that are hence 
empirically less informative than features that occur in a small fraction of the training corpus.

Consider a document containing 100 words wherein the word cat appears 3 times. The term 
frequency (i.e., tf) for cat is then (3 / 100) = 0.03. Now, assume we have 10 million documents 
and the word cat appears in one thousand of these. Then, the inverse document frequency 
(i.e., idf) is calculated as log(10,000,000 / 1,000) = 4. Thus, the Tf-idf weight is the product 
of these quantities: 0.03 * 4 = 0.12.

https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
scikit-learn includes several variants of the NAIVE BAYES classifier; the one most suitable for 
word counts is the multinomial variant


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""



from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize   
import re
import pandas as pd
from sklearn.pipeline import Pipeline
import numpy as np

stop_words = set(stopwords.words('english')) 


data=pd.read_excel('/users/josh.flori/desktop/t.xlsx')

# removed stop words comments, I found performance to be different when I just relied on sklearn to remove them
comments=[[" ".join([w.lower() for w in word_tokenize(re.sub(r'[^A-Za-z]',' ',data['comment'][i].split("------------------------------")[0].replace("\n",""))) if w not in stop_words])][0] for i in range(len(data['comment']))]

# test/train split
X_train, X_test, y_train, y_test = train_test_split(comments, data['yes_no'])



def sample_many(text_classifier,n):
    """ The purpose of this function is to repeat the training process many times on many different train/test splits to ensure nothing weird 
        manifested in any of the splits. This is mostly because the dataset I am using is so small, I like doing these sanity checks."""
    total=[]
    for i in range(n):
        # n is the number of times you want to repeat the classification on n different train/test splits
        X_train, X_test, y_train, y_test = train_test_split(comments, data['yes_no'])
        # fit a new model on new splits
        text_classifier.fit(X_train, y_train) 
        predicted=text_classifier.predict(X_test)
        score=np.mean(predicted == y_test)
        print(score)
        total.append(score)
    print("\n")
    print(np.mean(total))
    print(np.median(total))



""""""""""""""""""""""""
"""" CREATE PIPELINE """
""""""""""""""""""""""""
# create a pipeline, which is an easy way to create features and build a model.
# the following uses the CountVectorizer
text_classifier = Pipeline([
     ('vect', CountVectorizer()),
     ('tfidf', TfidfTransformer()),
     ('clf', MultinomialNB()), ])

text_classifier.fit(X_train, y_train) 
predicted=text_classifier.predict(X_test)
np.mean(predicted == y_test)







###########################################
###########################################
               EXTRA STUFF                
###########################################
###########################################
""" included for educational purposes, the following is just a longhand way of doing things without the pipeline..."""
count_vect = CountVectorizer()
# get tfidf information off of basic count information i guess
X_train_tfidf = TfidfTransformer().fit_transform(count_vect.fit_transform(X_train))
# fit the classifier.
classifier = MultinomialNB().fit(X_train_tfidf, y_train)
predicted = classifier.predict(X_test)
accuracy=[predicted[i][0]==y_test.tolist()[i] for i in range(len(predicted))].count(True)/len(predicted)


# if you want to know what the feature vectors actually look like in X_train_tfidf, one way to view that data is as follows.
# feature_vectors_list will hold features for the data like this: [[X_train[0]],[X_train[1]],[X_train[2]],[X_train[3]]...]
# and the length of each sublist X_train[i] will be equal to the vocabulary size, in this case it should be len(set(comments))
# it would look something like this [[0.0, 0.57, 0.0, 0.57, 0.57], where each entry corresponds to a word in the vocabulary and the number 
# is the calculated td-if for that word in that training example.
feature_vectors_list = []
for i in X_train_tfidf:
    feature_vectors_list.append(list(i.A[0]))






