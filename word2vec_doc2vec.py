# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 15:10:11 2016

@author: jaykim
"""

from gensim.models.word2vec import Word2Vec

model = Word2Vec.load_word2vec_format('vectors.txt', binary=False) #C text format
model = Word2Vec.load_word2vec_format('vectors.bin', binary=True) #C binary format



from gensim.models.word2vec import Word2Vec

model = Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary = True)

model.most_similar(positive=['woman', 'king'], negative=['man'], topn=5)


# [(u'queen', 0.711819589138031),
#  (u'monarch', 0.618967592716217),
#  (u'princess', 0.5902432799339294),
#  (u'crown_prince', 0.5499461889266968),
#  (u'prince', 0.5377323031425476)]


model.most_similar(positive = ['biggest', 'small'], negative=['big'], topn = 5)


# [(u'smallest', 0.6086569428443909),
#  (u'largest', 0.6007465720176697),
#  (u'tiny', 0.5387299656867981),
#  (u'large', 0.456944078207016),
#  (u'minuscule', 0.43401968479156494)]


model.most_similar(positive=['ate', 'speak'], negative=['eat'], topn=5)

# [(u'spoke', 0.6965223550796509),
#  (u'speaking', 0.6261293292045593),
#  (u'conversed', 0.5754593014717102),
#  (u'spoken', 0.570488452911377),
#  (u'speaks', 0.5630602240562439)]


import numpy as np

with open('food_words.txt', 'r') as infile:
    food_words = infile.readlines()

with open('sports_words.txt', 'r') as infile:
    sports_words = infile.readlines()
    
with open('weather_words.txt', 'r') as infile:
    weather_words = infile.readlines()

def getWordVecs(words):
    vecs = [ ] 
    for word in words:
        word = word.replace('\n', '')
        try:
            vecs.append(model[word].reshape((1, 300)))
        except KeyError:
            continue
    vecs = np.concatenate(vecs)
    return = np.array(vecs, dtype = 'float') #TSNE expects float type values

food_vecs = getWordVecs(food_words)
sports_vecs = getWordVecs(sports_words)
weather_vecs = getWordVecs(weather_words)


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

ts = TSNE(2)
reduced_vecs = ts.fit_transform(np.concatenate((food_vecs, sports_vecs, weather_vecs)))

#color points by word group to see if Word2Vec can separate them
for i in range(len(reduced_vecs)):
    if i < len(food_vecs):
        #food words colored blue
        color = 'b'
    elif i >= len(food_vecs) and i < (len(food_vecs) + len(sports_vecs)):
        # sports words colored red
        color = 'r'
    else: 
        # weather words colored green
        color = 'g'
    plt.plot(reduced_vecs[i, 0], reduced_vecs[i, 1], marker='o', color=color, marketsize = 8)


from sklearn.cross_validation import train_test_split
from gensim.models.word2vec import Word2Vec

with open('twitter_data/pos_tweets.txt', 'r') as infile:
    pos_tweets = infile.readlines()

with open('twitter_data/neg_tweets.txt', 'r') as infile:
    neg_tweets = infile.readlines()


# use 1 for positive sentiment, 0 for negative
y = np.concatenate((np.ones(len(pos_tweets)), np.zeros(len(neg_tweets))))

x_train, x_test, y_train, y_test = train_test_split(np.concatenate((pos_tweets, neg_tweets)), y, test_size=0.2)

# Do some very minor text preprocessing
def cleanText(corpus):
    corpus = [z.lower().replace('\n' '').split() for z in corpus]
    return corpus

x_train = cleanText(x_train)
x_test = cleanText(x_test)

n_dim = 300
#Initialize model and build vocab
imdb_w2v = Word2Vec(size=n_dim, min_count=10)
imdb_w2v.build_vocab(x_train)

#Train the model over train_reviews (this may take several minutes)
imdb_w2v.train(x_train)

# build word vector for training set by using the average value of all word vectors in the tweet, then scale
def buildWordVector(text, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in text:
        try:
            vec += imdb_w2v[word].resahpe((1, size))
            count += 1.
        except KeyError:
            contioue 
    if count != 0:
        vec /= count
    return vec 
    

from sklearn.preprocessing import scale
train_vecs = np.concatenate([buildWordVector(z, n_dim) for z in x_train])
train_vecs = scale(train_vecs)

#Train word2vec on test tweets
imdb_w2v.train(x_test)

#Build test tweet vectors then scale
test_vecs = np.concatenate([buildWordVector(z, n_dim) for z in x_test])
test_vecs = scale(test_vecs)

#Use classification algorithm (i.e., Stochastic Logistic Regression) on training set, then assess model performance on test set
from sklearn.linear_model import SGDClassifier

lr = SGDClassifier(loss='log', penalty='l1')
lr.fit(train_vecs, y_train)

print 'Test Accuracy: %.2f' %lr.scores(test_vecs, y_test)

#Create ROC curve
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

pred_probas = lr.predict_proba(test_vecs)[:,1]

fpr, tpr,_ = roc_curve(y_test, pred_probas)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='area = %.2f' %roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.legend(loc='lower right')

plt.show()


from NNet import NuralNet

nnet = NeuralNet(100, learn_rate=1e-1, penalty = 1e-8)
maxiter = 1000
batch = 150 
_ = nnet.fit(train_vecs, y_train, fine_tune=False, maxiter=maxiter, SGD=True, batch=batch, rho=0.9)

print 'Test Accuracy: %.f'%nnet.score(test_vecs, y_test)



import gensim

LabeledSentence = gensim.models.doc2vec.LabeledSentence 

from sklearn.cross_validation import train_test_split
import numpy as np

with open('IMDB_data/pos.txt', 'r') as infile:
    pos_reviews = infile.readlines()

with open('IMDB_data/neg.txt', 'r') as infile:
    neg_reviews = infile.readlines()
    
with open('IMDB_data/unsup.txt', 'r') as infile:
    unsup_reviews = infile.readlines()

#use 1 for positive sentiment, 0 for negative
y = np.concatenate((np.ones(len(pos_reviews)), np.zeros(len(neg_reviews))))

x_train, x_test, y_train, y_test = train_test_split(np.concatenate((pos_reviews, neg_reviews)), y, test_size=0.2)

#Do some very minor text preprocessing
def cleanText(corpus):
    punctuation = """.,?!:;(){}[]"""
    corpus = [z.lower().replace('\n','') for z in corpus]
    corpus = [z.replace('<br />', ' ') for z in corpus]
    
    #treat puctuation as individual words 
    for c in punctuation:
        corpus = [z.replace(c, ' %s '%c) for z in corpus]
    corpus = [z.split() for z in corpus]
    return corpus 

x_train = cleanText(x_train)
x_test = cleanText(x_test)
unsup_reviews = cleanText(unsup_reviews) 

#Gensim's Doc2Vec implementation requires each document/paragraph to have a label associated with it. 
#We do this by using the LabeledSentence method. The format will be "TRAIN_i" or "TEST_i" where "i" is
#a dummy index of the review.
def labelizeReviews(reviews, label_type):
    labelized = [ ]
    for i, v in enumerate(reviews):
        label = '%s_%s'%(label_type, i)
        labelized.append(LabeledSentence(v, [label]))
    return labelized

x_train = labelizeReviews(x_train, 'TRAIN')
x_test = labelizeReviews(x_test, 'TEST')
unsup_reviews = labelizeReviews(unsup_reviews, 'UNSUP')



import random

size = 400

#instantiate our DM and DBOW models
model_dm = gensim.models.Doc2Vec(min_count=1, window=10, size=size, sample=1e-3, negative=5, workers=3)
model_dbow = gensim.models.Doc2Vec(min_count=1, window=10, size=size, sample=1e-3, negative=5, dm=0, workers=3)

# build vocab over all reviews
model_dm.build_vocab(np.concatenate((x_train, x_test, unsup_reviews)))
model_dbow.build_vocab(np.concatenate((x_train, x_test, unsup_reviews)))

#We pass through the data set multiple times, shuffling the training reviews each time to improve accuracy.
all_train_reviews = np.concatenate((x_train, unsup_reviews))
for epoch in range(10):
    perm = np.random.permutation(all_train_reviews.shape[0])
    model_dm.train(all_train_reviews[perm])
    model_dbow.train(all_train_reviews[perm])

# Construct vectors for test reviews
test_vecs_dm = getVecs(model_dm, x_test, size)
test_vecs_dbow = getVecs(model_dbow, x_test, size)

test_vecs = np.hstack((test_vecs_dm, test_vecs_dbow))


from sklearn.linear_model import SGDClassifier

lr = SGDClassifier(loss='log', penalty='l1')
lr.fit(train_vecs, y_train)

print 'Test Accuracy: %.2f' %lr.score(test_vecs, y_test)



# Create ROC curve
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

pred_probas = lr.predict_proba(test_vecs)[:,1]

fpr, tpr,_ = roc_curve(y_test, pred_probas)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='area = %.2f' %roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.legend(loc='lower right')

plt.show()


from NNet import NeuralNet

nnet = NeuralNet(50, learn_rate=1e-2)
maxiter = 500
batch = 150
_ = nnet.fit(train_vecs, y_train, fine_tune=False, maxiter = maxiter, SGD=True, batch=batch, rho = 0.9)

print 'Test Accuracy: %.2f'%nnet.score(test_vecs, y_test)

