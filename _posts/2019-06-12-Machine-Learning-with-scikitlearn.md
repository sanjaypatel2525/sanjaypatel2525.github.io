---
layout: post
title: Machine Learning with Python
date: 2018-12-14 18:57:06.000000000 -05:00
type: post
parent_id: '0'
published: true
password: ''
status: publish
categories: [Machine Learning Notes]
tags: [Machine Learning]
randomImage: '30'
meta:
  _edit_last: '1'
author:
  login: sanjaypatel2525
  email: sanjaypatel2525@gmail.com
  display_name: sanjaypatel2525
  first_name: ''
  last_name: ''
---


### Loading dataset.
Digits
{% highlight Python %}
from sklearn.datasets import load_digits, load_iris,load_diabetes,load_boston
import matplotlib.pyplot as plt 
digits.load_digits()
plt.imshow(digits.images[0], cmap=plt.cm.gray_r)
plt.show()
{% endhighlight %}
* If fit is called again it will scrap the old classifier and refit. 

### Models
{% highlight Python %}
linear_model.LinearRegression()
Lasso() # Sparse or L1 
Ridge() # L2
LogisticRegression(penalty='l2',C=1e5) , large C less regulazrization.
svm.SVC(kernel='linear') # Support vector classification
svm.SVR(kernel='linear')  # Support vector regression
{% endhighlight  %}

### Scores
{% highlight Python  %}
from sklearn.metrics import accuracy_score
accuracy_score(pipe.predict(X_test), y_test) 
clf.score() # gives score on classifier directly

k_fold = KFold(n_splits=5)
score_array = cross_val_score(svc, X_digits, y_digits, cv=k_fold,scoring='precision_macro')
avg_score = np.mean(score_array)

from sklearn.model_selection import cross_validate
result = cross_validate(linearClassifier, X, y) # same cross_val_score,default to 5
result['test_score']

clf = GridSearchCV(estimator=svc, param_grid=dict(C=Cs),n_jobs=-1) # njob is number of processor, -1 is all
clf.best_score_  
clf.best_estimator_.C 
{% endhighlight  %}

### Grouping
{% highlight Python  %}
k_means = cluster.KMeans(n_clusters=3)
k_means.fit(X_iris)
k_means.labels_ # it is predicted label array of trained data.

agglo = cluster.FeatureAgglomeration(connectivity=connectivity,n_clusters=32)
agglo.fit(X)
X_reduced = agglo.transform(X)

# PCA
pca = decomposition.PCA()
print(pca.explained_variance_)  
[  2.18565811e+00   1.19346747e+00   8.43026679e-32]
# As we can see, only the 2 first components are useful
pca.n_components = 2
X_reduced = pca.fit_transform(X)
X_reduced.shape

pca = PCA(n_components=n_components, svd_solver='randomized',
          whiten=True).fit(X_train)
X_train_pca = pca.transform(X_train)

#pipeline
pipe = sklearn.pipeline(steps=[('pca', pca), ('logistic', logistic)])
param_grid = {
    'pca__n_components': [5, 15, 30, 45, 64],  # here use the name given in pipeline 'pca' or 'logistic'
    'logistic__C': np.logspace(-4, 4, 4),
}
search = GridSearchCV(pipe, param_grid, n_jobs=-1)

from sklearn.pipeline import make_pipeline  #another form of pipe with automatic names
pipe = make_pipeline(
    StandardScaler(),
    LogisticRegression(random_state=0)
)

scipy.sparse.csr.csr_matrix #is used in place of numpy for sparse matrix

{% endhighlight %}

**vector quantization** - Once kmean is run, the seleceted points can be used as infomrational points which in the sense compresses the data to those points. 
**Agglomerative or feature agglomeration.** -Compress feature to smaller dimensions using clustering algorithems. 

### Text Addons
{% highlight Python  %}
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
count_vect = CountVectorizer()  # creates bag of word and counts the occurance of word.
X_train_counts = count_vect.fit_transform(twenty_train.data)
count_vect.vocabulary_.get(u'algorithm') #4690

tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts) # TF- means within single document, each word count is divide by total word count, so total remains 1. IDF - means common word in all document are given less attention.
X_train_tf = tf_transformer.transform(X_train_counts)

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)
docs_new = ['God is love', 'OpenGL on the GPU is fast']
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts) #it's not fit again, as already fit in training.
predicted = clf.predict(X_new_tfidf)

{% endhighlight  %}

### Transformer
{% highlight Python  %}
# Remove missing value
from sklearn.preprocessing import LabelEncoder, OneHotEncoder # used for encoding categorical data
from sklearn.preprocessing import StandardScaler # used for feature scaling

#First set unwanted value to np.NaN
X.replace({999.0 : np.NaN}, inplace=True)
indicator = MissingIndicator(missing_values=np.NaN)
indicator = indicator.fit_transform(X)
indicator = pd.DataFrame(indicator, columns=['m1', 'm3'])

# Impute or fill missing values
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='mean') #mean, most_frequent, median and constant 
imp.fit_transform(X)
#Or in Pandas just
X.fillna(X.mean(), inplace=True)

imputer = SimpleImputer(missing_values=np.nan, strategy='mean') imputer = imputer.fit(X[:, 1:])
X[:, 1:] = imputer.transform(X[:, 1:])

from sklearn.preprocessing import StandardScaler

#Encoder
#Onehotencoder(required integer data,2d array), labelBinarizer(doesn't required integer data, 2d array), labelEncoder(encode data to integer, 1d array)
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

{% endhighlight  %}



