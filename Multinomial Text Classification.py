# https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html

""" MODELS USED: NAIVE BAYES, SVM"""
# scikit-learn includes several variants of this classifier; the one most suitable for word counts is the multinomial variant:



"""Fortunately, most values in X will be zeros since for a given document less than 
a few thousand distinct words will be used. For this reason we say that bags of words 
are typically high-dimensional sparse datasets. We can save a lot of memory by only 
storing the non-zero parts of the feature vectors in memory.

scipy.sparse matrices are data structures that do exactly this, and scikit-learn 
has built-in support for these structures.

Text preprocessing, tokenizing and filtering of stopwords are all included in CountVectorizer, 
which builds a dictionary of features and transforms documents to feature vectors:




--TD-IDF--
Consider a document containing 100 words wherein the word cat appears 3 times. The term 
frequency (i.e., tf) for cat is then (3 / 100) = 0.03. Now, assume we have 10 million documents 
and the word cat appears in one thousand of these. Then, the inverse document frequency 
(i.e., idf) is calculated as log(10,000,000 / 1,000) = 4. Thus, the Tf-idf weight is the product 
of these quantities: 0.03 * 4 = 0.12.
"""



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


data=pd.read_excel('/users/josh.flori/path/text_data.xlsx')
# removed stop words comments
comments=[[" ".join([w.lower() for w in word_tokenize(re.sub(r'[^A-Za-z]',' ',data['comment'][i].split("------------------------------")[0].replace("\n",""))) if w not in stop_words])][0] for i in range(len(data['comment']))]

# test/train split
X_train, X_test, y_train, y_test = train_test_split(comments, data['yes_no'])


# CountVectorizer builds a dictionary of features and transforms documents to feature vectors. Once fitted, the vectorizer has built a dictionary of feature indices:  The index value of a word in the vocabulary is linked to its frequency in the whole training corpus. Supposedly it removes stopwords but i found performance improved when i removed stopwords via ntlk
# Occurrence count is a good start but there is an issue: longer documents will have higher average count values than shorter documents, even though they might talk about the same topics.
# To avoid these potential discrepancies it suffices to divide the number of occurrences of each word in a document by the total number of words in the document: these new features are called tf for Term Frequencies.
# Another refinement on top of tf is to downscale weights for words that occur in many documents in the corpus and are therefore less informative than those that occur only in a smaller portion of the corpus.
# This downscaling is called tf–idf for “Term Frequency times Inverse Document Frequency”.
"""# the following is just a longhand way of doing things without the pipeline...."""
                  ------------------------------------------------------
                  ------------------------------------------------------                  
                  count_vect = CountVectorizer()              
                  # get tfidf information off of basic count information i guess
                  X_train_tfidf = TfidfTransformer().fit_transform(count_vect.fit_transform(X_train))
                  # fit the classifier.
                  classifier = MultinomialNB().fit(X_train_tfidf, y_train)
                  predicted = classifier.predict(X_test)
                  accuracy=[predicted[i][0]==y_test.tolist()[i] for i in range(len(predicted))].count(True)/len(predicted)
                  ------------------------------------------------------

def sample_many(text_classifier,n):
    total=[]
    for i in range(n):
        X_train, X_test, y_train, y_test = train_test_split(comments, data['yes_no'])
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
#----------------------
""""""""""""""""""""""""
"""" MULTINOMIAL NB  """
""""""""""""""""""""""""                  

text_classifier = Pipeline([
     ('vect', CountVectorizer()),
     ('tfidf', TfidfTransformer()),
     ('clf', MultinomialNB()), ])

text_classifier.fit(X_train, y_train) 
predicted=text_classifier.predict(X_test)
np.mean(predicted == y_test)