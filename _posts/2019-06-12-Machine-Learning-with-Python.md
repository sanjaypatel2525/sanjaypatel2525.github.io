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
<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>

Exploratory data analysis, data visualization, and clustering, classification, regression and model performance evaluation.
Python has various libraries such as, Numpy, matplotlib, scipy, sckitlearn for data science and data analysis.

Popular machine learning techniques classification, regression, recommendation and clustering.

Python has many data mining algorithm implementation such as linear regression, logistic regression, naive bayes, k-mean, k nearest neighbor and random forest.

Data science, machine learning and artificial intelligence are few trending topic these. Data mining and Bayesian analysis has increased demand of machine learning.

Programming which learns and improves with experience.  Learning means recognizing and understanding the data and taking informed decision based on the provided data points. The algorithm are developed such that they build the knowledge from data and past experience by applying statical science, probability, logic, mathematical optimization, reinforcement learning and control theory.

Here are few application of machine learning.
* Vision processing
* language processing
* Forecasting number ex, stock market, weather
* pattern recognition
* games
* data mining
* expert system
* robotics

Steps in machine learning.
* Defining problem
* Preparing data
* Evaluating algorithm
* improving results
* presenting results

### Wait, What's the difference between data scientist and ML Engineer.
* Data science guy is more of studious kid who's job is to see data remove error, find patterns, analyzing data, creating graph etc.
* ML engineer is not a data guy, he gets the data and starts applying available algorithms or create prototypes to get meaningful output from data. Normally data scientist request ML engineer to create a algorithm/program based on the prototype they have.

Python libraries
* numpy - N-dimensional array object.
* pandas - dataframes manipulation
* matplotlib/searborn - plotting graphs, data visualization
* scikit-learn - algorithm for data analysis and data mining.

Machine learning has following the common type machine learning algorithms.
* Supervised
* Unsupervised
* Semi-Supervised
* Reinforced learning


## Supervise learning
Ex face recognition, speech recognition, recommendation based on history, forecasting etc. These data are fed to algorithm and tuned to provide expected data points as output.

In supervise learning the input data comes as labeled data so we know the result and we try to propose solution which takes inputs data set and tries to predict correct label, later we try to improve algorithm so that the accuracy of prediction is increased. Once the algorithm reaches satisfactory mark it can be used with real data. It is based on labeled sample and the output is know with learning data.
There are many supervised algorithm examples: **linear regression, logistic regression, support vector machine SVM, Naive bayes classifier**.

**Classification**- Classification is dividing the data into set, categorizing it or anything related to segregating data. It is done when you have complete data, so it is done after.  
**Prediction/Regression** is about guessing/predicting about the input data. In this you work on training example and train the model and later use the model to predict on new data. This is further division in specific type of machine leaning such as supervised, unsupervised, reinforced. **It works on continuos values and continuos value have order, less than a number or more than a number, not like discrete, no order, only set of values**

## Unsupervised learning.
It used to detect anomalies, outlier, something which is odd then normal, fraud, defective equipment, or group with similarities. In here we don't get the labeled data. It is also called unlabeled learning, the algorithm tries to find pattern, structure, anomaly from the given data.
Since here we do not know feature which can act as data classification points so unsupervised learning tries to group them in different groups based on data underlying patterns in optimum way. Most of the time unsupervised learning tries to find similarities and cluster them. Some examples are Kmean, random forest, hierarchal clustering etc.  
**Association & Clustering** - If you buy bread then system shows jam as well, this kind comes under association. Clustering is again grouping but without training data. 

## Semi supervised.
Some data are labeled and some are not, unlabeled are used in unsupervised learning and labeled are used for testing and fine tuning. It saves cost when it is difficult to get full labeled data.

## Reinforced learning.
Here learning is improved based on real time data feedback. System adjust itself based on learning data. Ex are self driving car, chess Alpha Go. Unlike supervised it works only on three feedback- happy, unhappy and neutral. Supervised tries to find right hyper parameters for model and updates these based on some function based on output.

The goal of machine learning is to reduce human effort but not creating the intelligence that piece goes into artificially intelligence which is superset of machine learning. It can evolve to go beyond human perception specializing in one given task.

![](/assets/2019-06-12-Machine-Learning-with-Python.png "Courtsey https://becominghuman.ai/cheat-sheets-for-ai-neural-networks-machine-learning-deep-learning-big-data-678c51b4b463"){: .lazyload}
Or more related to.
![](/assets/2019-06-12-Machine-Learning-with-Python2.jpg){: .lazyload}

## Data preparation and preprocessing
Let's talk a bit about **standardization and normalization**.
### Normalization
Data can in be sometime in big number or may be sometime in very small number, there are multiple technique to scale data to readable or to our needed limits. For example. 0<x<1 scaling. Any data can be scalded from 0 to 1 in this its easy to know lower limit and upper limit and we can easily perceptually how big one data in this scale.  
**Min-Max Normalization** - To fit the data in defined boundary, let say data has 134567 min and 136878 max and we want to see it in scale of 0 to 1, or in scale of 1000 to 10000 etc.
{% raw %}$$B= \frac{A-min(A)}{max(A)-min(A)}\cdot(D-C) + C$${% endraw %}, it lays the data A<sub>i</sub> from set of A, from C to D.  
**Decimal Scaling** - Divide or multiple with 10<sup>n</sup> to bring the at scale of decimal. For example
10,30, 4000 will become 0.001, 0.03, 0.4 after deviling with 10<sup>4</sup>.  
**Standard Deviation** - {% raw %}$$\sigma = \sqrt\frac{\sum _{i=1}^{N}(x_i-\mu)^2}{N}$${% endraw %}

**Standardization or z-score** Is form of normalization where mean is kept to 0 and standard deviation to 1. In graph the start and end of data limits has to have same distance from y axis.{% raw %}$$z = \frac{x_i-\mu}{\sigma}$${% endraw %}


## Algorithms.
### Linear regression
Regression tries to find the least cost continuos function based on given input such that, the aim to minimize the cost. The cost is cost of given training set combined. Cost is also called loss. There are many loss function out which here is one **mean squared error( L2 Loss function)**.
It better then sum of absolute error as sum of absolute error can have ambiguous result while squared will penalize larger distance and find exact half way.
![](/assets/2019-06-12-Machine-Learning-with-Python39.JPG){: .lazyload}
$$Cost= J(\theta_0,\theta_1) = \frac{1}{2m}\sum_{i=0}^m(h_\theta(x^i)-y^i)^2\\
h_\theta(x)=y=\theta_0+\theta_1x$$
Where theta are independent variable and hyperparameter, m is number of training data, x is input data and y is output data, h is expected output data. Our aim is to find minimum cost for given input variables. 
With one variable.![](/assets/2019-06-12-Machine-Learning-with-Python3.JPG){: .lazyload}
With two variable. ![](/assets/2019-06-12-Machine-Learning-with-Python4.JPG){: .lazyload}
**Gradient Descent** Gives change in input to reach minimum cost.
$$\theta _{jnew}=\theta_j-\alpha\frac{\delta}{\delta\theta_j}J(\theta_0,\theta_1)$$
Where derivative of J gives the slope which is always -1 to 1 and this slope will be calculated for each variable. This slope calculates new value for variable θ. α defines the jump either to decrease or to increase this variable. 

### Multivariate Linear regression
$$h_θ=θ_0+θ_1x_1+θ_2x_2..+θ_nx_n,
x=\begin{bmatrix}x_0\\x_1\\..\\x_n\end{bmatrix},
θ=\begin{bmatrix}θ_1\\θ_1\\..\\θ_n\end{bmatrix}
h_θ(x)=θ^Tx$$  
$$\theta _{jnew}=\theta_j-\alpha\frac{1}{m}\sum_{i=1}^m(h_θ(x^i)-y^i)x_j^i$$

**Feature Scaling** - If the variable are not in same scale they tend to make gradient jump and slows down the process of reaching to minimum. So there are many feature scaling. 
* Standardization - Replace by z-score, Normally used in normally distributed set. $$x'=\frac{x-\bar x}{\sigma}$$
* Mean Normalization - Can variate from -1 to 1, used in Principal component analysis. $$x'=\frac{x-\bar x}{max(x)-min(x)}$$
* Min-Max Scaling - from 0 to 1. $$x'=\frac{x-min(x)}{max(x)-min(x)}$$

### Normal equation to get Optimal value
Normal equation gets you the minimum value of θ directly, for example you know the curve of the cost function. Then you can solve for minimum value for θ.
If θ is quadratic. $$J(θ)=aθ^2+bθ+c = 0$$ will give you minimum value for θ. Same goes for multiple variable keep one variable constant each time. 
m - Training examples
n - Variables. 
$$x=\begin{bmatrix}x_0\\x_1\\..\\x_n\end{bmatrix},x^i=\begin{bmatrix}x_0^i\\x_1^i\\..\\x_n^i\end{bmatrix},
X= \begin{bmatrix}[x^1]^T\\ [x^2]^T\\ [x^m]^T\end{bmatrix},
min\,value\,of\,θ=(X^TX)^{-1}X^Ty$$
**Advantage**, you don't need to &alpha; and it does not take lot of iteration like gradient descend while **disadvantage** is has complexity of  $$O(n^3)$$ and difficult to work with nonconvertible equation $$(X^TX)^{-1}$$, though it can be solved by removing redundant feature, removing feature and regularization.

### Logistic Regression -
Logistic function or called sigmoid function, it maps the value to in between 0 < h(x) < 1. Hence can be used in probability prediction. Probability of y being 1 at given value of x when θ is hyperparameter. θ changes the width of sigmoid function hence affect the y over over x. 
$$h_θ(x)=g(z)=g(θ^Tx)= \frac{1}{1+e^{θ^Tx}}=P(y=1|x:θ)$$

A good explanation here. https://towardsdatascience.com/logistic-regression-detailed-overview-46c4da4303bc
https://towardsdatascience.com/logistic-regression-detailed-overview-46c4da4303bc

* Logistic is some what categorized as classification algorithm but is is continuos. 

**Decision Boundary**- Kind of line, curved line, plane which divides the data into sets. It is used to define the boundary of classification data. For example a line dividing area in 2D plain.
![](/assets/2019-06-12-Machine-Learning-with-Python5.JPG){: .lazyload}
* Gradient decent can't be applied on logistic regression as it is non linear (not straight line), the cost function will not be mapped as normal convex curve but it will have multiple local minima in it. So we use another cost function which give convex graph for J(θ).
$$Cost(h_θ,y)=\begin{cases}-log(h_θ(x))\text{ if y=1}\\-log(1-h_θ(x))\text{ if y=0}\end{cases}$$
* Ok **biggest question** here is how do you select fitting a cost function. If you are researcher or very good at maths you might guess a fitting graph (logistic(classification) or linear(continuos prediction)), and then you guess what cost function will be good for my model. There are lot of cost function you select any one and try if your machine is performing well or not. But some are pretty standard cost function for given fitting graph for ex mean square for linear, the one above for sigmoid etc.

![](/assets/2019-06-12-Machine-Learning-with-Python6.JPG){: .lazyload}
Looks same as linear regression, but they differ by $$h_θ(x)$$.
"Conjugate gradient", "BFGS", and "L-BFGS" are alternative of gradient descend and provide better optimization over cost.

### Multi-class classification.
In logistic we use to have two set yes/no and each event has some probability based on that we use to say this event belong to yes or no group. That probability is derived by function $$h_θ(x)$$. Here we have many sets and each set will have its own $$h_θ^i(x)$$. We calculate y's probability in eahc set and assign this to the set which has max probability.
![](/assets/2019-06-12-Machine-Learning-with-Python7.JPG){: .lazyload}

### Underfitting & overfitting. 
If you increase the variable to map all the training data then there are chances the test data may not fall close to the curve. This is overfitting so to reduce this,
* Sometime reduce the parameter
* Sometime keep it but regularize. 
Increase the λ in regularize some variables, so even small change in variable will change cost to much, to keep cost low these variable should be small. λ is regularization parameter
$$min_θ\frac{1}{2m}∑_{i=1}^m(h_θ(x^{(i)})−y^{(i)})^2 +1000⋅θ_3^2 ​ +1000⋅θ_4^2,
=​min_θ\frac{1}{2m}∑_{i=1}^m(h_θ(x^{(i)})−y^{(i)})^2 +λ\sum_{j=1}^nθ_j^2$$

## Neural network (Andrew Ng style)
$$a^1=x=\begin{bmatrix}x_0=1\\x_1\\..\\x_{S_j}\end{bmatrix},
θ^1=\begin{bmatrix}θ_{10}&θ_{11}&..&θ_{1S_j+1}\\θ_{20}&θ_{21}&..&θ_{2S_j+1}\\..\\θ_{S_j0}&θ_{S_j1}&..&θ_{S_jS_j+1}\end{bmatrix},
h_θ(x)=a^2=g(z^2)=g(θ^{1}a^{1})$$ 
* There is no $$z^1$$
Deriving from logistic regression. But compared to logistic it has K output in multi class. 
![](/assets/2019-06-12-Machine-Learning-with-Python8.JPG){: .lazyload}

**Backward proportion** - We know when there is hight cost or more error in each K output nodes, then we need to fix each θ in previous layers. This is done through backward propagation.
Let δ is error in Kth node of L, $$δ_k^l = a_k^l-y_k$$ now calculate δ at L-1 layer. This is done through derivative of 
$$(a^{l-1})'=g'(z^{l-1})=a^{l-1}.(1-a^{l-1})$$. 
δ is applied at nodes we need to adjust our θ matrix. So we put another term Δ which is proportionate to δ
$$Δ^{(l)}_{i,j} := Δ^{(l)}_{i,j} + a_j^{(l)} δ_i^{(l+1)}$$
$$D^{l}_{i,j}=\frac{1}{m}(Δ^{l}_{i,j}+λθ^l_{i,j},\,if\,j≠0,
D^{l}_{i,j}=\frac{1}{m}Δ^{l}_{i,j}\,if\,j=0$$
![](/assets/2019-06-12-Machine-Learning-with-Python10.JPG){: .lazyload}
 
**Gradient Checking** - Compare the delta changed value to the output to the the change we got from Backward propagation. Gradient Checking is time consuming that is the reason we do not use it in place of backward propagation.
![](/assets/2019-06-12-Machine-Learning-with-Python9.JPG){: .lazyload}.

## Evaluating Hypothesis.
Multiple things we can tweak to fix our hypothesis when we start.
* Increase training set
* Increase decrease parameters
* Increase decrees generalization λ.
* Try polynomial equations over parameters.
![](/assets/2019-06-12-Machine-Learning-with-Python11.JPG){: .lazyload}.
Some people **divide training** set in training set, validation set to pick **lowest cost polynomial function** and test set to check **generalization variation** on new data. 

### Bias and variance.
As you increase d polynomial degree the validation error will decrease at it will not be generalized and will have high variance.
![](/assets/2019-06-12-Machine-Learning-with-Python1.png){: .lazyload}

**Effect of λ**
![](/assets/2019-06-12-Machine-Learning-with-Python12.JPG){: .lazyload}

**Effect of training data**
In high bias(less parameterized) model training after some time is of no use, while in high variance training will improve the learning rate and decrease the Cross Validation(CV) cost.
![](/assets/2019-06-12-Machine-Learning-with-Python13.JPG){: .lazyload}

* Always start with small model and keep drawing learning curve and calculating CV cost. Try different approach on small model to find out if more parameter needed if more generalization needed.

**Precision & Recall evaluation** - Sometime you get the training whose output is skewed, in the sense hight percentage on one class. With these kind of data algorithms can't learn well and show low precision and low recall characteristics. Evaluating Precision and recall give idea wether the algorithm is heading in right direction or not.
![](/assets/2019-06-12-Machine-Learning-with-Python14.JPG){: .lazyload}
![](/assets/2019-06-12-Machine-Learning-with-Python16.JPG){: .lazyload}
Higher F score mean a better algorithm it take value between 0 and 1.

## Support Vector Machine
One of the famous Supervised learning. 
![](/assets/2019-06-12-Machine-Learning-with-Python17.JPG){: .lazyload}
In image we get rid of $$\frac{1}{m}$$ and λ. $$\frac{1}{m}$$ is constant so doesn't affect cost function in minimization and intuition is in cost function keep bigger B in order generalize any parameter, that you can do either by increasing λ r remove lambda put C and decrease C, both are same thing.

SVM is also called **Large Margin Classifier**.

Think $$\theta^T x^i$$ as vector multiplication and you can say what this multiplication gives projection of &theta; over x.
![](/assets/2019-06-12-Machine-Learning-with-Python18.JPG){: .lazyload}
![](/assets/2019-06-12-Machine-Learning-with-Python19.JPG){: .lazyload}
In above example our aim is to get maximum of p so that θ can be small. With bigger p there is bigger distance of dataset from the classifier line and will have bigger margin. 

**Kernel** Above we were discussing are devisor which was linear, in some situation the plotted points are mixed and requires nonlinear curve to divide or enclosing circle. 
![](/assets/2019-06-12-Machine-Learning-with-Python20.JPG){: .lazyload}
Here we take some points and mark them as territory and calculate the closeness/similarities of each dataset from these territories/points. Mark these points as *l* and calculate closeness/similarities vector. 
$$f_i = similarity(x,l^i)$$
Let say closeness is defined by.
$$f^i = similarity(x,l^i) = exp(-\frac{||x-l^i||^2}{2\sigma^2})$$.
When σ is big the peak is distributed(more generalized) and when it is small peek is thin(more variant).
![](/assets/2019-06-12-Machine-Learning-with-Python21.JPG){: .lazyload}
![](/assets/2019-06-12-Machine-Learning-with-Python22.JPG){: .lazyload}
So we have two kernel linear and gaussian kernel, linear for straight line divisor an gaussian for non-linear. When m is very large compared to n then using SVN with Gaussian kernel is good. Otherwise use linear kernel.

### K-Mean 
K-mean is unsupervised clustering algorithm, Here you tak randomly take k clusters and each cluster has a centroid $$μ_k$$. We calculate $$c^i$$ for each $$x^i$$ which states cluster for $$x^i$$, which cluster x belongs to by calculating distance with each cluster centroid.
$$c^i = min||x^i-\mu_k||^2$$
This c maps all the input to some cluster and later each cluster's mean is adjusted with average of x from that cluster.
$$\mu_{c^i}=\frac{(x^a+x^b_...)}{\text{count of x in }\mu}$$

**Cost Function** - We keep calculating K-mean until out cost stops decreasing. Cost is calculate by squared sum of distance of x from there cluster mean.
$$J(c^1,..,c^m,\mu_1,..,\mu_K)= \frac{1}{m}\sum_{i=1}{m}||x^i-\mu_{c^i}||^2$$



**Optimization** - Initialize mean of each cluster to random value of inputs and calculate cost, the lowest cost is selected in the end as optimized solution.

**Deciding number of Clusters K** - Use elbow approach, keep increasing K from 2 to n draw the graph, the cost keeps on decreasing but at some point we can see elbow, the rate of cost decrease has reduce significantly. Or if there is no such elbow select whatever suites you.
![](/assets/2019-06-12-Machine-Learning-with-Python23.JPG){: .lazyload}

### Data compression.
Compress the data from 2D to 1D or 3D to 2D or nD to mD m< n etc, to save the memory and computation overheads.
Correlated feature can be mapped together using a function or even directly so reduce multiple feature in one.

**PCA(Squared Projection/minimum distance Error)** - We try to find a line, where 2D points are projected and line should have minimum projection error. It is different to linear regression as linear regression tries to find out a line which tries minimizes the distance between actual y and projected h(x).  
![](/assets/2019-06-12-Machine-Learning-with-Python24.JPG){: .lazyload}
![](/assets/2019-06-12-Machine-Learning-with-Python25.JPG){: .lazyload}
In above diagram you can see how K is selected so that k variance by total variance is 99.99% retained.
* PCS speed up, use less memory and if k=1 ,2 ,3 then easy to visualize. This does not generalize the model as it doesn't not consider the y in supervised learning while generalization parameter &lambda; does.

### Anomaly detection Algorithm.
![](/assets/2019-06-12-Machine-Learning-with-Python26.JPG){: .lazyload}
![](/assets/2019-06-12-Machine-Learning-with-Python27.JPG){: .lazyload}
Using training set we try to define a gaussian curve on each feature and with new x having n feature we try to find probability of this x to be fitted to gaussian curve, if it is not a good fit it will have very low probability output.
**Choosing feature**
* Create new feature if existing feature can''t find anatole
* Create features out of another features so that the value is either very high or very low.
* Transform feature so that it falls in gaussian curve, apply log or srqrt or any other root.

**Multivariate Gaussian Distribution**
Sometime we need Multivariate Gaussian because in previous we use to find out p(x<sub>1</sub>),p(x<sub>2</sub>) etc. and this use to ignore there coherent property, In case we need to look how the combined gaussian looks we calculate p(x).
![](/assets/2019-06-12-Machine-Learning-with-Python28.JPG){: .lazyload}
In simple term how it is different from normal previous gaussian is the previous oen can not take diagonal ellipse form while this one can. In that case covariance matrix '&sum;' all **non diagonal fields will be 0**, while in this case non diagonal field will have some weight.
![](/assets/2019-06-12-Machine-Learning-with-Python29.JPG){: .lazyload}

### Content based recommendation
Let' remember the linear regression again, It tries to map feature of training set features= x<sub>1</sub>,x<sub>2</sub>...x<sub>n</sub>, input data =x<sup>1</sup>,x<sup>2</sup>...x<sup>m</sup>  to the output of training set with tweaking parameters as &theta;
Now take n<sub>m</sub>  as movies and n<sub>u</sub> as users. We need to fit &theta; for each user to movies input based movies feature. Since there are n<sub>u</sub> users hence there will &theta;<sup>j</sup> one for each user.
![](/assets/2019-06-12-Machine-Learning-with-Python34.JPG){: .lazyload}
![](/assets/2019-06-12-Machine-Learning-with-Python30.JPG){: .lazyload}

### Collaborative Filter Algorithm
The soul remains same θ and x,m normals we try to find θ to fit y for given x. So that said x is defined at least. But in last example x was some features of movies and how we can find those features, so we start small. With give θ provided by users (user rates some movies and also tell there inclination romance, action etc), with this information we calculate x. This calculated x is used to predict θ  and it goes in loop x and θ  both predict each one after one. 
But better than going one by one lets find everything together.
![](/assets/2019-06-12-Machine-Learning-with-Python31.JPG){: .lazyload}

### Recommendation Algorithm
If user is watching one movie suggest him another movies. This is normally found by distance between two movies features x<sup>1</sup> ($$x_1^1,x_2^1..x_n^1$$). The distance is calculated by $$||x^1-x^2||^2$$.
What if there is new user and we need to suggest him a movie. The θ<sup>i</sup> will be 0 for him as the model doesn't have any parameter which affect θ for this new user. 
![](/assets/2019-06-12-Machine-Learning-with-Python32.JPG){: .lazyload}

### Stochastic Gradient descent
The normal  Gradient descent is called batch  Gradient descent, where we have to find mean square of all the input data on each iteration of gradient descent update. In case if m is in million then each θ update will have to calculate million records before making updates. In Stochastic Gradient descent we update θ based on last $$x^i,y^i$$.
![](/assets/2019-06-12-Machine-Learning-with-Python33.JPG){: .lazyload}

### Improving performance.
**Artificial Synthesize** the data. 
**Ceiling Analysis** prepare one liner chart to see if you provide truth data to component how is the accuracy is increase and with this you can find out what needs to be fine tuned more rather then wasting time on other component.

#Read this 
https://www.altexsoft.com/blog/datascience/machine-learning-project-structure-stages-roles-and-tools/
https://towardsdatascience.com/building-package-for-machine-learning-project-in-python-3fc16f541693
https://towardsdatascience.com/the-ultimate-guide-to-data-cleaning-3969843991d4
https://medium.com/omarelgabrys-blog/statistics-probability-exploratory-data-analysis-714f361b43d1#48f7

# Udacity Intro ML

### Bayes. 
Bayes formula tells you how to calculate P(A|B) when P(B) and P(B|A) is given. Just say you are interested to know what if I changes terms P(B|A) to P(A|B)
Here is an example. 
![](/assets/2019-06-12-Machine-Learning-with-Python35.JPG){: .lazyload}
### Naive Bayes
Based on all words(evidences) calculate final probability of being some label. It is classification algorithm but doesn't consider order so that why it called naive bayes.
![](/assets/2019-06-12-Machine-Learning-with-Python36.JPG){: .lazyload}

### SVM
SVM large margin classifier, first agenda is classify then create a large margined.
SVC(C=1.0, kernel=’rbf’, degree=3, gamma=’auto_deprecated’, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape=’ovr’, random_state=None)[source]
* Gamma - defines whether a the boundary has more local points effect or it consider far points as well. Small value is nearby point, 
* C - Defines the regularization parameters. lower value mean more regularized.
* Kernel  - Defines what type of feature relations you  are going ot use.‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable.
  
**Compared to Naive bayes** - On more feature SVM slows down also kernel overlap issue is not there in Naive Bayes. SVM works well in non linear domain as it tries to capture relation between feature(kernel or function).

### Decision tree. 
Creates a tree internally on each feature with value less then or more then, keeps on splitting till min_samples_split is reach, minimum is 2 . But when you increase it, it works as regularization parameter.
class sklearn.tree.DecisionTreeClassifier(criterion=’gini’, splitter=’best’, max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, presort=False)[source]
![](/assets/2019-06-12-Machine-Learning-with-Python42.JPG){: .lazyload}
![](/assets/2019-06-12-Machine-Learning-with-Python37.JPG){: .lazyload}
![](/assets/2019-06-12-Machine-Learning-with-Python38.JPG){: .lazyload}
In above diagram, we calculate entropy of parent and entropy at its children, subtract it and we get how much this classification gained us knowledge. Entropy is maximum possible impurity(non purity), if there can be only 1 output then chance of having impurity is 0 hence 0 entropy.
**Information Gain** - We find information gain on all features using above formula, it helps us determine which feature is most suitable for split to get highest information gain.
Decision Tree works good on classifying and can create big tree, but it always prone to overfit. 

## NLTK
### Stemmer
Consolidating the words to single meaning. 
![](/assets/2019-06-12-Machine-Learning-with-Python40.JPG){: .lazyload}

### TFIDF
Term frequency in a sentence and inverse document in frequency in a corpora or whole document. If word occurs more in sentence will have more term frequency, But if it occurs more in document the attention to this word has to be less. 
https://kavita-ganesan.com/tfidftransformer-tfidfvectorizer-usage-differences/#.XdLPzVdKgdU 
tfidf_vectorizer=TfidfVectorizer(use_idf=True) 
tfidf_vectorizer_vectors=tfidf_vectorizer.fit_transform(docs)

### PCA 
Principal component analysis, tries to reduce the dimension by retaining most of the information. Let's say there are two axis x(area) and y(bedroom), it is directly proportional and create a slanted line in graph of x,y so we can change our origin and slop of base axis. This forms an new dimension which is created using other two dimension.
In algorithm we don't try to find out one by one, we just dump all in algo, PCS tries to find most suitable by ranking and that calls it first PC line, second PC line and so on but all these PCA (new dimension) will be perpendicular to each other. 
![](/assets/2019-06-12-Machine-Learning-with-Python41.JPG){: .lazyload}

### Eigenfaces (PCA of facial data)
Using overall training set we try to reduce facial data dimensionality to smaller dimensions. 

### GridSearchCV
Use sklearn GridSearchCV for let sklearn figure out best possible parameter for specific algorithm from given parameters.

![](/assets/2019-06-12-Machine-Learning-with-Python43.JPG){: .lazyload}

### Helper libraries in Python
TextBlog - sentiment analysis inbuilt.
LighFM - Mixed recommender system
CSV - To read csv
PIL(Pillow) - For images
URLLib - Download files
ZIPFiles - To unzip/zip files
os - For os related tasks
TPOT - BUilt over sklearn to find best algorithm and parameter

### Recommender system.
Collaborative - What other people like
Content based - what you like.