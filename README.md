# Naive Bayes Classifiers
https://www.cs.waikato.ac.nz/ml/publications/2004/kibriya_et_al_cr.pdf

#### Variants:

MultinomialNB

TWCNB (transformed weight-normalized complement naive bayes)

TFIDFNB (term frequency times inverse document frequency naive bayes)

-----------------------------------------------------------------------

## MultinomalNB
![alt text](https://imgur.com/6No9siw.png)

### Phrased as (ignoring the log): 
"The probability of some class k given some train/test example x<sup>1</sup>, is proportional to the probability of that class k out of all classes<sup>2</sup>, multiplied by the joint probabilities of the words in that example<sup>3</sup>. 

<sup>1</sup> In other words, if you are classifying text into one of two classes, this is asking, "for some new piece of text, what's the probability that it is class k?"

<sup>2</sup> This will be equal to 1/(num_classes)

<sup>3</sup> The probability of each word is in base form equal to the count of occurances of that word in that class divided by the total number of words in that class. That would be equal to (5)/(8) in the picture below. But smoothing operators can be added where 1 is a constant, and 6 is the vocabulary size. 

So for the picture below, the probability of class C for example d5 is equal to the probability of the class C (1/2) * p(chinese|c) * p(chinese|c) * p(chinese|c) * p(tokyo|c) * p(japan|c). Pretty straightforward.


Per https://www.youtube.com/watch?v=km2LoOpdB3A the reason we use a proportional sign is because we are not actually computing the probability, but the numerator of the probability. The denominator would be the same in both P(class1|example1) and P(class2|example1) so it can just be left out.

![alt text](https://imgur.com/7DvTvYI.png)

This is basically equivilent to bayes theorem as shown. Where in the image below, P(class|data) is the same as saying P(class s|d5), but then the upper terms are reversed from the example above, such that P(class) is equal to 1/num_classes and P(data|class) is the joint probability of all data in the class, in our case words.
  
  
  
![alt_text](https://imgur.com/a8BftNc.png)

some additional information
![alt_text](https://imgur.com/Lbm314V.png)
