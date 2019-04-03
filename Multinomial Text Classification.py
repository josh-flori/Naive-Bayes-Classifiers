""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                                       NOTES                                        

CountVectorizer builds a dictionary of features and transforms documents to feature vectors. 
Once fitted, the vectorizer has built a dictionary of feature indices:  The index value of a word 
in the vocabulary is linked to its frequency in the whole training corpus.

Text preprocessing, tokenizing and filtering of stopwords are all included in CountVectorizer, 
which builds a dictionary of features and transforms documents to feature vectors. It also
lowers the words. <<<<<<<<<

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


Every sklearn's transform's fit() just calculates the parameters (e.g. 𝜇 and 𝜎 in case of 
StandardScaler) and saves them as an internal objects state. Afterwards, you can call its transform() 
method to apply the transformation to a particular set of examples.

fit_transform() joins these two steps and is used for the initial fitting of parameters on the 
training set 𝑥, but it also returns a transformed 𝑥′. Internally, it just calls first fit() and then 
transform() on the same data.

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
comments=[[" ".join([w for w in word_tokenize(re.sub(r'[^A-Za-z]',' ',data['comment'][i].split("------------------------------")[0].replace("\n",""))) if w not in stop_words])][0] for i in range(len(data['comment']))]

# You can also choose to not remove stopwords and leave that up to sklean
comments=[[" ".join([w for w in word_tokenize(re.sub(r'[^A-Za-z]',' ',data['comment'][i].split("------------------------------")[0].replace("\n","")))])][0] for i in range(len(data['comment']))]


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
# accuracy
np.mean(predicted == y_test)




"""""""""""""""""""""""""""""""""""
"""" REPEAT N TIMES (OPTIONAL) """"
"""""""""""""""""""""""""""""""""""
sample_many(text_classifier,n)







###########################################
###########################################
               EXTRA STUFF                
###########################################
###########################################
""" included for educational purposes, the following is just a longhand way of doing things without the pipeline..."""
count_vect = CountVectorizer()
tfidf_transformer = TfidfTransformer()

X_train_counts = count_vect.fit_transform(X_train)
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)


X_train_tf = tf_transformer.transform(X_train_counts)
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)


# get tfidf information off of basic count information i guess
X_train_tfidf = TfidfTransformer().fit_transform(count_vect.fit_transform(X_train))





# fit the classifier.
classifier = MultinomialNB().fit(X_train_tfidf, y_train)
predicted = classifier.predict(X_test)
accuracy=[predicted[i][0]==y_test.tolist()[i] for i in range(len(predicted))].count(True)/len(predicted)



from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)


docs_new = ['God is love', 'OpenGL on the GPU is fast']
X_new_counts = count_vect.transform(x)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

predicted = clf.predict(X_new_tfidf)




    
    
    
----------------------    
    
# The first line below instantiates a CountVectorizer. The second line will actually change the value of that count_vect. Count_vect inherents the original train vocabulary and the model parameters will be learned against it. It must be passed to the new test data to create word counts on that data. From both train and test, it must be passed into a tfidf transfomer to generate tfidf data.
# CountVectorizer filters stopwords, lowers words and tokenizes words.
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(x_train)

# Run this:
for i in X_train_counts:
    print(list(i.A[0]))
# And observe that you get a result like:
# [4, 1...]
# [1, 5...]
# [4, 1...]
# Where rows correspond to rows/training_examples in X_train and length of each row is = (total vocab - removed stopwords). The integers are the counts of that vocab word in that training example.

# To take count data and get tfidf data from it, run:
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tf_transformer.fit_transform(X_train_counts)

# And run this:
for i in X_train_tf:
    print(list(i.A[0]))
# And observe that you get a result like:
# [0.970, 0.242...]
# [0.196, 0.980...]
# [0.970, 0.242...]
# where rows correspond to rows in x, and length is = (total vocab - removed stopwords). The numbers are the td-idf for that word.



# Fit the classifier.
classifier = MultinomialNB().fit(X_train_tfidf, y_train)
predicted = classifier.predict(X_test)



count_vect.transform(x_test)
# any new words in x_test not in x_train will not be included in the output (confirmed!)
# all that will happen is that for each test example you will be returned an array the length of the train vocab, and in each spot of that array is the count of that train vocab word in the x_test example... which i guess makes sense. so you will want to run a test that checks how many words in new data were not in vocabulary. presumably not much but im sure at some point you will see a lot of new words on the basis of different threads have different keywords.

# so the important components here are count_vect which inherents the original vocabulary and which the parameters will be learned against, it must be passed to the new test data to create counts, then from the counts you need to create tdidf information using
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
# tfidf_transformer will create an output identical in size to count_vect.transform(x), it will just output tfidf instead of counts.

# you can run your model, then generate prediction on new data by doing:
X_test_counts = count_vect.transform(x_test)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)

# Then you can process that through the fitted model to get predictions
classifier = MultinomialNB().fit(X_train_tfidf, y_train)
predicted = classifier.predict(X_test)






