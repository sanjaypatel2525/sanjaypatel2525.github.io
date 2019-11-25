---
layout: post
title: Machine Learning Neural Network.
date: 2018-12-13 18:57:06.000000000 -05:00
type: post
parent_id: '0'
published: true
password: ''
status: publish
categories: [Machine Learning Notes]
tags: [Machine Learning]
randomImage: '33'
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

Neural network set of network with input, hidden and output neurons/nodes. Each neuron is **activation function** (for ex gradient descend) and each synapse/link have some weights. Every node maintains error metric and coefficient are adjusted to lower the error metric using back propagation. **Bias** are some constant values at neuron.

**Ok all above is fine but why?**  
![](/assets/2019-06-12-Machine-Learning-Basic-Maths22.JPG)
Let think of example, what number is there in below image. It is blurred not accurate but it is 2. Every human handwriting will create different image of 2 probably but human understand it is 2. How machine can know it is 2. Actually it doesn't know it, it says the probability of this image being 2 is let say 90%. And we write the code which can take pixel of this image as input and generate possible results with probability. How much a pixel is lit can be i/p for next layer and affect how much next layer neuron are lit, in the end if algorithm is good it will lit the output neuron which is marked for 2 with some probability.


**activation function** - Typically have Non-linear, continuos differentiable and fixed range.  

**Loss or Cost Function** - Defines accuracy of prediction with given neural network model. 

**Optimization Algorithms**- //TODO
## Forward propagation
Let say we have n layers and let's pick two layer, each layer will have some activation function, Superscript number is layer number and subscript number is neuron number on that layer from top to bottom, $$a_k^n$$ will n<sup>th</sup>  layer and k<sup>th</sup> node/neural at that level.
* $$w^L$$ is weight at level L
* $$b^L$$ is bias at level L
* $$a^{L-1}$$ is X or input for level L and output of level L-1 which is activation function applied on z<sup>L</sup>.
* $$z^L = w^la^{L-1}+b^L$$
* C is cost function. $$C_0=(a^L-y)^2$$
* y is expected output
* $$\sigma'$$ is derivative of activation function wrt z<sup>L</sup>

So the formula to derive $$a_k^n$$ will be
{% raw %}
$$a^1=\sigma(Wa^0+b)\\
\begin{bmatrix}a_1^n\\a_2^n\\..\\a_k^n\end{bmatrix}= \sigma\left(\begin{bmatrix}W\end{bmatrix}^{k_{n-1}\times k_{n}}\begin{bmatrix}a_1^{n-1}\\a_2^{n-1}\\..\\a_k^{n-1}\end{bmatrix}+\begin{bmatrix}B\end{bmatrix}^{{k_n}\times 1}\right)$$
{% endraw %}
## Back propagation
Goal is to adjust the weights so that each overall cost is minimized.
Example of 1 node at each level. 

![](/assets/2019-06-12-Machine-Learning-Basic-Maths20.JPG)
Calculate change in C wrt to W<sub>L</sub>. $$\frac{\delta C_0}{\delta w^L}=\frac{\delta z^L}{\delta w^L}\frac{\delta a_L}{\delta z^L}\frac{\delta C_0}{\delta a^L}\\
= a^{(L-1)}.\sigma'(z^L).2(a^L-y)$$
For k number of training examples. $$C_1,C_2..C_k$$ will be the cost function and nudging L level weight w<sub>L</sub> will be $$\frac{\delta C}{\delta w^L}=\frac{1}{n}\sum_{k=0}^{n-1}\frac{\delta C_k}{\delta w^L}$$
We saw weight now we need talk about changing b<sup>L</sup> and a<sup>L-1</sup>
$$ \nabla C = \begin{bmatrix}\frac{\delta C}{\delta w^1}&\frac{\delta C}{\delta a^{b^1}}&\frac{\delta C}{\delta a^{L-1}}\\
\frac{\delta C}{\delta w^2}&\frac{\delta C}{\delta a^{b^2}}&\frac{\delta C}{\delta a^{L-2}}\\
...&...&...\\
\frac{\delta C}{\delta w^L}&\frac{\delta C}{\delta a^{b^L}}&\frac{\delta C}{\delta a^0}
\end{bmatrix} $$
So C wrt to $$b^L$$ and $$a^{L-1}$$ is as follows.
$$\frac{\delta C_0}{\delta b^L}=\frac{\delta z^L}{\delta b^L}\frac{\delta a_L}{\delta z^L}\frac{\delta C_0}{\delta a^L}\\
= 1.\sigma'(z^L).2(a^L-y)\\\\
\frac{\delta C_0}{\delta a^{L-1}}=\frac{\delta z^L}{\delta a^{L-1}}\frac{\delta a_L}{\delta z^L}\frac{\delta C_0}{\delta a^L}\\
= w^L.\sigma'(z^L).2(a^L-y)$$
Now let take k node at L-1 level and j nodes at L level. At this level $$\nabla C$$ changes at seems complex but it is not. 
$$\Delta C =\eta \nabla C$$ Where $$\nabla c$$ is negative small change, gradient descend and $$\eta$$ is learning rate or step size. Keeping too small will slow down and keeping big will overshoot. Later update w and b by these small changes.
![](/assets/2019-06-12-Machine-Learning-Basic-Maths21.JPG)


Commutatively we can say $$\delta_j^L=\frac{\delta C}{\delta z^l_j}$$. If this value big that mean changes at this z will make big impact to lower down the C rather then other lower derivative.

**stochastic gradient descent** - If there are large training set, the learning might take time so we take a small set.

### Activation Function
There are few substantial activation function which we use most of the times, sigmoid, tanh(scaled sigmoid), relu, leaky ReLu. 
**Step** - On of off, 0 or 1, decide based on threshold. This is not analog so turning off few neuron completely will make them passive which we don't want. 

**Linear** - Change in proportionate to w and $$a^n$$. If the all the layer have lennar function, then logically combined function will be linear and it will be dumb.

**Sigmoid** - $$f(x)=\frac{1}{1+e^{-2x}}$$. It fits the a from range $$[-\infty,+\infty]$$ to [0,1]. Gradient is result hungry if it is in between [-2,2] beyond this gradient is very slow and stops learning.

**tanh** - $$tanh(x)= 2sigmoid-1$$, scales to [-1,1] beyond it gradient is slow.

**ReLu** - A(x) = max(0,x). Activation value Less then 0 will be 0. Makes calculation faster but makes half network passive as they are always off. 

**Leaky ReLu** - Less then 0 is kept very small but not 0, y = 0.0x. Makes all nodes active. Solves ReLu problem. There other variant, **ELU**, less then 0 will be exponential $$\alpha(e^z-1)$$ 

Thanks to 3Blue1Brown videos on neural network, makes visualization so easy.
{% raw %}<iframe width="420" height="315" src="https://www.youtube.com/watch?v=tIeHLnjs5U8" frameborder="0" allowfullscreen></iframe>{% endraw %}

### Increase efficiency.
**Cross Entropy Function** - (Not same as probability distribution cross entropy, do not get confused). In place quadratic cost function (y-a)<sup>2</sup> use cross entropy function to calculate change required for weigh and biases. Learning rate doesn't slow down as quadratic cost function.
$$C=−1n∑x[ylna+(1−y)ln(1−a)]$$

**SoftMax and loglikelyhood** - In place of 0 to 1 output probability on each output layer(sigmoid activation), if we say combined output layer probability should be 1(Softmax activation). Then the data is more comparable. Here cost function used is -log a, in case of output close to 1 less change is needed and in 0 more change is required.
 $$\sum_j a_j^L=\frac{\sum_j e^{z_j^L}}{\sum_k e^{z_k^L}}\\
 C=-log\;a_y^L$$

**overfitting or overtraining** - the epoch from where you dont see much learning after that.

**Bias & Variance Tradeoff** - Bias is difference between average predicted value to true value. High bias Pays very little attention to training set and over simplifies the model(**underfitting**). For ex. Model devices is linear function while the actual need was non linear function. Variance tells about the spread of the data. High variance over learn the training set and doesn't generalize it (**overfitting**). It picks the outliers/noise and noise also in its knowledge which is overfiting of the data.  High parameter increase high variance and low parameter increase high bias, we have to trade off in selecting right parameter.
$${\displaystyle \operatorname {E} {\Big [}{\big (}y-{\hat {f}}(x){\big )}^{2}{\Big ]}={\Big (}\operatorname {Bias} {\big [}{\hat {f}}(x){\big ]}{\Big )}^{2}+\operatorname {Var} {\big [}{\hat {f}}(x){\big ]}+\sigma ^{2}}\\
=(f-E[\hat f])^2+Var[y]+Var[\hat y]\\
=(f-E[\hat f])^2+E[\varepsilon^2]+E[(E[\hat f]-\hat f])^2]$$

**Detect underfit and overfit** - Divide training data in training and test data and use test data to get the accuracy of model. 
**Fixing underfit and overfit**
* Cross-validation - Divide data in splits and train and test. For ex K-Fold, divide data in K and keep increasing the training set, keep decreasing the test set. 
* Train with relevant data- More data can be good, but if it noisy it is issue so train with relevant data.
* Remove feature - Remove irrelevant features. Rubber duck debugging.
* Early stop - More training overfits the data sometime so know when to stop.
* Regularization - Make model simpler sometime, prune decision tree, dropout neural network, penalty parameter.
* Ensembling - Multiple model learn separately and combine them in the end to smooth it out. Bagging and boosting are example. Bagging start with complex model then smooths it out while boosting start with weak learner models and form a complex model.  

**Weight decay Regularization- L2 or Ridge Regularization** - Add a extra term regularization term, $$C=C_0+\frac{\lambda}{2n}\sum_ww^2\\
\frac{\delta C}{\delta w} = \frac{\lambda}{n}w$$. It helped in overcoming overfitting issue and increasing accuracy, also saved us from local minima.


**Weight decay Regularization-L1 or Lasso Regularization (LAD - Least Absolute deviation)** - Since 

**Sparsity** - Defines how much element in vector or matrix are zero, more zero means more sparsity. L1 is more sparse.

**Built-In feature Selection** - Since L1 is sparse it brings down the wait of non used parameters which automatically bring more weighted parameters.

**Dropout** - We select set of hidden neuron and turn them off and repeat the machine learning for small batches, later average it out. It remove the overfitting of data as different network will pick different things and on average it will span out better. 

* Increase in training data increases the accuracy. We can increase training data by skewing image, rotating image etc. 
* Initialize weight right, Normally we use normal/gaussian distribution in that case mean is 0 and SD is 1 but since these normally distributed weights will make z 0 and any change in weight will not impact cost hence learning will be very slow. Use $$\frac{1}{\sqrt n}$$.

**Derive Hyper parameter fast**
* Work with simple output prediction, such as 0,1 yes no etc.
* Start with simple network
* Take small batches. 
* Adjust lambda in proportion to reduces batch size.
* $$\eta$$, start with big and small see where it starts oscillating, it should not overshoot.
* Keep monitoring, stop early. no-improvement-in-ten etc.



# Neural Network Andrews Ng way
![](/assets/2019-06-12-2020-06-12-Machine-Learning-Neural-Network1.JPG)
![](/assets/2019-06-12-2020-06-12-Machine-Learning-Neural-Network2.JPG)
Vectorization takes advantage of SIMD (Single instruction Multiple data)
### Gradient Descend vectorization
![](/assets/2019-06-12-2020-06-12-Machine-Learning-Neural-Network9.JPG)
![](/assets/2019-06-12-2020-06-12-Machine-Learning-Neural-Network3.JPG)
![](/assets/2019-06-12-2020-06-12-Machine-Learning-Neural-Network4.JPG)
![](/assets/2019-06-12-2020-06-12-Machine-Learning-Neural-Network5.JPG)

* In NN input layer is not counted as layer, so we start from $$a^[0]$$ which is input layer. If you see 3 layers including input and output, it will be called 2 layer NN.

**Multilayer neural network vectorized form**
![](/assets/2019-06-12-2020-06-12-Machine-Learning-Neural-Network6.JPG)
![](/assets/2019-06-12-2020-06-12-Machine-Learning-Neural-Network7.JPG)
![](/assets/2019-06-12-2020-06-12-Machine-Learning-Neural-Network8.JPG)

**Derivative of activation function**
$$sigmoid = g(z) = a \;then\; g'(z)= 1-a \\
tanh = g(z)=tanh(z) = a \;then\;g'(z) = 1-a^2\\
relu = g(z)=max(0,z) = a \;then\;g'(z) = \begin{cases} 0\;when\;z<0\\1\;when\;z>0\end{cases}\\
leaky relu = g(z)=max(0.01z,z) = a \;then\;g'(z) = \begin{cases} 0.01\;when\;z<0\\1\;when\;z>0\end{cases}$$

![](/assets/2019-06-12-2020-06-12-Machine-Learning-Neural-Network10.JPG)

**Keep track of dimensions**
![](/assets/2019-06-12-2020-06-12-Machine-Learning-Neural-Network11.JPG)

**Deep network calculations**
![](/assets/2019-06-12-2020-06-12-Machine-Learning-Neural-Network12.JPG)

**Backward and forward propagation formulas**
![](/assets/2019-06-12-2020-06-12-Machine-Learning-Neural-Network13.JPG)

## Fine-tune NN
Earlier ML used to have 70, 30 ratio to train and test set. Now a day due to big data the data set has increase so train set goes around 99% remaining is dev and test set. Dev test cross validation data for developer to find out right ML algorithm. Dev and test set should be from same distribution set, if not dev and test wouldn't be same.

### Bias and Variance.
Training error gives idea about bias, high error is high bias. Dev error -training error gives variance, high difference high variance. In NN we do not have tradeoff situation, as it was there in ML algorithms. NN provide different ways to reduce only bias or only variance.
Fix Bias : Increase neural network size by increasing layer or increasing nodes, Or train longer, Or choose better NN architecture.
Fix variance: More data or Regularize, or better NN architecture. 

### Regularization 
L1 - Absolute value of weights
L2(weight decay) - Squared value of weights
![](/assets/2019-06-12-2020-06-12-Machine-Learning-Neural-Network14.JPG)


### Dropout
Dropout, on off different nodes at different inputs tends to generalize the network. This is also one of the regularization technique.
**Inverted Dropout**
![](/assets/2019-06-12-2020-06-12-Machine-Learning-Neural-Network15.JPG)

### Vanishing and exploding Gradient descend.
Many times gradient becomes to big or too small which just diverges the cost function. To overcome this issue there are multiple solution.
**Random Weights initialization based on Activation function**
Based on number of input node distribute weight to sum up to 1, which make each weight 1/n. Relu works better with 2 in place of 1.
![](/assets/2019-06-12-2020-06-12-Machine-Learning-Neural-Network16.JPG)

**Gradient Checking** - Sometime while coding we would want to know weather our gradients calculation are correct in code. To do so we merge all W's, b's,dW's and db's to single dimension array call &theta; and d&theta;. Later calculate approx d&theta; by nudging &#949; a bit. In the end compare the difference between these approx and model's calculated derivatives.
![](/assets/2019-06-12-2020-06-12-Machine-Learning-Neural-Network17.JPG)

**Grad check Do's and Don'ts**
* Do it only once, do not do it each training iterations.
* In case difference is big compare component wise, for ex $$dW_L[i,j]$$ to  $$dW_L[i,j]_{aprox}$$
* Doesn't work with dropout.

### Optimization.
**Mini Batch gradient descend** - Create batch, calculate and apply weight changes and iterate over all the batch. Repeat this till you get to minimum. It adds up noise in cost function curve. It is fastest as it usage mini batch and vectorization both.

**Stochastic G D** - Mini batch size is 1. It is more noisy then mini batch. It doesn't use vectorization much hence it is slow.

**Batch GD** - Here batch size is m, full train dataset.

**Fit mini batch to CPU GPU memory to run faster**, such 64, 128, 246,512, beyond this is rare for now. 

**Exponential weighted Avg** - Averaging current with previous values with some proportion to previous values.
![](/assets/2019-06-12-2020-06-12-Machine-Learning-Neural-Network18.JPG)
![](/assets/2019-06-12-2020-06-12-Machine-Learning-Neural-Network19.JPG)


**Bias Correction** - In Exponential weighted average, initial average is biased as it starts from zero, to fix we use a different formula. 
$$v_{t+1} = \frac{v_t}{1-\beta^t}$$

**Apply EWA to Gradient Descend** - This somewhat resolves issue of noise due small batch size, also helps with the issue of overshooting with higher learning rate. 
![](/assets/2019-06-12-2020-06-12-Machine-Learning-Neural-Network20.JPG)

**RMS prop** - Root mean square, same as momentum dampening.
![](/assets/2019-06-12-2020-06-12-Machine-Learning-Neural-Network22.JPG)

**Adam Optimization** - It is combination of both RMS and EWA. 

**Learning rate decay** - Once model start reaching to minimum vale the learning rate can be reduced so that it converges well. There are multiple formula's to decay learning rate based on epoch. For ex. 
$$\alpha=\frac{1}{1+decay\_rate*epcoh\_num}$$

### Local optima problem.
Multidimensional space local optima is normally not a problem as dimension increase the chances of reaching local minima are very low. For ex 20000 feature will have probability 1/20000. So it is actually a plateaus, the thing put back on horse to sit on. It will always have other directions which can still minimize.

### Batch Norm
Sometime it is good to normalize z's or a'z of each layer.
We calculate tilda z, as normalized form of z, and rescale it with &beta; and &gamma;
![](/assets/2019-06-12-2020-06-12-Machine-Learning-Neural-Network23.JPG)
&beta; is constant and can be eliminated as we already have rescaled the z.
![](/assets/2019-06-12-2020-06-12-Machine-Learning-Neural-Network24.JPG)
Batch norm intuition is about, hidden layer don't need to bother about changing inputs due to earlier layer learning, with z and  a norm hidden layer will learn based on a standard input, doesn't whatever initial layers have. It make hidden layers independent of initial layers.
* At test time, we use trained EWA of &mu; and &sigma; value.

## Softmax Classification.
When the output of NN has predefined classes, it becomes a problem of softmax where we calculate probability of classes. Hardmax just says 1 for one class and others as 0.
![](/assets/2019-06-12-2020-06-12-Machine-Learning-Neural-Network25.JPG)

### Single Number Evaluation matrix for Model.
![](/assets/2019-06-12-2020-06-12-Machine-Learning-Neural-Network26.JPG)

**F1 score** is harmonic mean(reduces effect of outlier average), **precision**(% of actual cats out of model recognized cats) and **recall**(% of correctly recognized cat out of actual cats) 

### Satisfying and optimizing metric
Sometime there is good precision of model but take a lot time, some has low precision but takes less time. Create  Satisfying criteria, what is max time be bare to run and with that find model which has highest precision.
![](/assets/2019-06-12-2020-06-12-Machine-Learning-Neural-Network27.JPG)

* Dev and test set should be from same distribution.
* We can omit test set but not advised.
* For really unwanted predicted result, penalize those result to increase the error percentage.
![](/assets/2019-06-12-2020-06-12-Machine-Learning-Neural-Network28.JPG)

## Bayes's Error
It's a upper limit defined by us before starting on any problem, We can't make machine more accurate they Bayes's error. For ex, faded image, even human give answer only 80% times right then machine can't have 100% accuracy so we define an upper limit for machine learning and it is Baye's error. Sometime It is same as human error but always above then human error.
![](/assets/2019-06-12-2020-06-12-Machine-Learning-Neural-Network29.JPG)

Avoidable Bias is gap between human error and training error.

* Carry out error analysis - In errors, classify errors and see which type of error is most and fix that.
* Random errors in training set is fine, but if it is systematic machine will learn on that error and create issues.

### Tuning Model using.
![](/assets/2019-06-12-2020-06-12-Machine-Learning-Neural-Network30.JPG)

### Transfer learning
Sometime you can use existing model with different problem such as cat detection to radiology or speech recognition to wake work. In this you just add remove last layer and append one or more extra layer.
![](/assets/2019-06-12-2020-06-12-Machine-Learning-Neural-Network31.JPG)

### Multitask learning
Autonomous driving one image can have multiple thing, signs, lights, pedestrian etc. All can stacked up in Y matrix and use bigger network to learn all thing together rather then one by one. It always works better then training individual neural network.
![](/assets/2019-06-12-2020-06-12-Machine-Learning-Neural-Network32.JPG)


### End to end deep learning
Big enough neural network with huge train data would suffice the for any complex x to y mapping. Some time when we less x to y mapping train data and we have more of intermediate data then we use component wise neural network model and here it works best, if you have huge data then end to end works best.
For ex, image to text detection, text to phrase detection, phrase to intent detection. This can be split into multiple small models or can be created in single big neural network, based on train set we decide which one to choose. 
![](/assets/2019-06-12-2020-06-12-Machine-Learning-Neural-Network33.JPG)


# Convolution network.
For images, you have million pixels in your images these, that mean million feature, implementing this in normal neural network will be very inefficient, so we convolute(complex) neural net.

### Edge detection.
Use 3X3 filter(Kernel) to find edges.
![](/assets/2019-06-12-2020-06-12-Machine-Learning-Neural-Network34.JPG)
![](/assets/2019-06-12-2020-06-12-Machine-Learning-Neural-Network35.JPG)
![](/assets/2019-06-12-2020-06-12-Machine-Learning-Neural-Network36.JPG)
* In above diagram you white to black is comes as white edge and black to white comes as dark edge, later you 30 and 10 that is because the edge has converted from white edge to back edge. If you transpose the filter matrix it becomes horizontal filter. Later image you see there are other filter which has some other properties, some are oriented filter(not vertical or horizontal but rotated to some degrees). Neural net learns to find best possible filter matrix.

### Padding
Since output becomes smaller then input we pad input pixel with 0 so that output becomes same size of input. Filter are normally of odd sized so padding should be full number. 
![](/assets/2019-06-12-2020-06-12-Machine-Learning-Neural-Network37.JPG)

### Strided Convolution.
Jump more then 1 step in applying filter. 
![](/assets/2019-06-12-2020-06-12-Machine-Learning-Neural-Network38.JPG)

**Maths, Cross-Correlation vs Convolution** - In maths all above work which we have done is cross-correlation, convolution involves one more step of filliping filter before applying.
![](/assets/2019-06-12-2020-06-12-Machine-Learning-Neural-Network39.JPG)

### Multiple channels(Third dimension ex RGB) and Multiple filters combines(ex Horizontal with vertical etc)
![](/assets/2019-06-12-2020-06-12-Machine-Learning-Neural-Network40.JPG)

### Notations for One layer CNN
Here we discuss about only one layer, consider different filters operation produces a different output nodes, in single filter you have 3 dimension data which is actually a flattened weight on 3*3*3 size vector.
![](/assets/2019-06-12-2020-06-12-Machine-Learning-Neural-Network41.JPG)

### Type of CNN layers
* Convolution Layer - Which we saw above
* Pooling Layer (Reducer, no parameter to learn)
* Fully connected layer.

### Pooling layer
These layer are the reducers, they reduce the size of each dimension by either max pooling (get max out of 1 block) or average (rarely used). It doesn't have any hyperparameter hence no learning of variables here. This is said to be used to pick one particular feature from block(Max) or average of block feature.
![](/assets/2019-06-12-2020-06-12-Machine-Learning-Neural-Network42.JPG)


### Simple CN (Ex Lenet-5)
* Normally the pattern is CL and PL few more CL and PL then some FC layers.
* H and W decreases while C(third dimension) increases
* PL has 0 parameter to lean, CL has less parameters, FCL has most parameters to learn.
![](/assets/2019-06-12-2020-06-12-Machine-Learning-Neural-Network43.JPG)

### Why convolution?
![](/assets/2019-06-12-2020-06-12-Machine-Learning-Neural-Network44.JPG)

### Case study.
Here we have three different CNN papers.
![](/assets/2019-06-12-2020-06-12-Machine-Learning-Neural-Network45.JPG)

### Res Net .
Residue net, it keeps the residue of previous layer by giving shortcut input to higher layers.
![](/assets/2019-06-12-2020-06-12-Machine-Learning-Neural-Network46.JPG)

**Why ResNets work,** It fights back to diminishing issue due to Weight Decay L2 regularization. If Weight diminishes in next layer, the input from previous layer are already mapped to third layer.
![](/assets/2019-06-12-2020-06-12-Machine-Learning-Neural-Network47.JPG)

Resnet has two type of blocks. First when the shortcut doesn't match the output dimension(convolution block) and when it matches (identity block)
![](/assets/2019-06-12-2020-06-12-Machine-Learning-Neural-Network87.JPG)

### 1 by 1 network
Polling can be used to reduce height and width, how about depth or channels, it is decrease by 1 by 1 filter.
![](/assets/2019-06-12-2020-06-12-Machine-Learning-Neural-Network48.JPG)

### Inception Layer
It usage 1 by 1 network concept to reduce the computation cost. Here is the comparison.
![](/assets/2019-06-12-2020-06-12-Machine-Learning-Neural-Network50.JPG)
Inception layer have multiple layer together and stacked up later. Combining multiple inception layer can work better also adding fully connected layer 

### Transfer learning
Take related model from github and replace last softmax layer with your classification softmax layer when you have very less data to train, in this case you will freeze calculation of previous layers and use pre calculated values , if you have more data to train, take github data start with that model and train whole model.
![](/assets/2019-06-12-2020-06-12-Machine-Learning-Neural-Network51.JPG)

## Object detection
Object detection is finding out object inside a picture, it more advance then the detecting what image is out or image classification.
![](/assets/2019-06-12-2020-06-12-Machine-Learning-Neural-Network52.JPG)

### Notation and Loss calculation for object detection.
bx, by are center for object, c1, c2,c3 will either have 0 or 1 which class it belong to, and Px is probability(mostly 1) of that object.
![](/assets/2019-06-12-2020-06-12-Machine-Learning-Neural-Network53.JPG)

### LandMark detection (Multiple points or shapes)
![](/assets/2019-06-12-2020-06-12-Machine-Learning-Neural-Network54.JPG)

### Sliding window with Convolution.
Sliding window you will have to crop many pictures, but convolution NN automatically does that with step as stride numbers.So all cropped images has run simultaneously and shared the computations
![](/assets/2019-06-12-2020-06-12-Machine-Learning-Neural-Network55.JPG)

### YOLO Algorithm
It runs very fast, it divides the image in grid and learn on weather the object is falls in box or not and it height/width with respect to box size. With these input data, YOLO starts to learn picking things.
![](/assets/2019-06-12-2020-06-12-Machine-Learning-Neural-Network56.JPG)
**What you should remember:**
* YOLO is a state-of-the-art object detection model that is fast and accurate
* It runs an input image through a CNN which outputs a 19x19x5x85 dimensional volume.
* The encoding can be seen as a grid where each of the 19x19 cells contains information about 5 boxes.
* You filter through all the boxes using non-max suppression. Specifically:
  * Score thresholding on the probability of detecting a class to keep only accurate (high probability) boxes
  * Intersection over Union (IoU) thresholding to eliminate overlapping boxes
Because training a YOLO model from randomly initialized weights is non-trivial and requires a large dataset as well as lot of computation, we used previously trained model parameters in this exercise. If you wish, you can also try fine-tuning the YOLO model with your own dataset, though this would be a fairly non-trivial exercise.

### IoU
![](/assets/2019-06-12-2020-06-12-Machine-Learning-Neural-Network57.JPG)

### Non Max suppression.
Model tend to detect multiple boxes for same object, then non max suppression picks one with highest probability and with this box find IoU with other boxes, if IoU of highest Px box and other box is grater then 0.5, then we discard box.
![](/assets/2019-06-12-2020-06-12-Machine-Learning-Neural-Network58.JPG)

### Multiple object over lap.
Create anchor boxes and increase y vector by maximum overlapping object that model should detect. Lets say y was 8 and we changed it to 8*2=16, it can detect two shape anchor boxes. 
![](/assets/2019-06-12-2020-06-12-Machine-Learning-Neural-Network59.JPG)
![](/assets/2019-06-12-2020-06-12-Machine-Learning-Neural-Network60.JPG)


## Face Recognition
Face verification is comparing image with another image, while recognition is finding out which person is this among thousands person.
**One Shot Learning** - These day system takes only one image of subject and starts recognizing it. So if you have only 1 image of subject and have to recognize its other images then we use One shot learning technique. It mainly work on distance between two given images. Same image distance will be 0 while different image distance will be more. Once is network is trained to pick different feature from face, it can be used for new subject image as well. Calculate feature vector of new person and save in DB for first time, later use this feature vector to judge weather provided image is of this person or not. 
&alpha; is margin, how much different can be accepted as same image. 
**Hard triplet** - For image distance learning, if you provide image subject(A), positive image(P) and negative (N), A and N are different people, then the distance will be definitely high and it will not help machine to learn. N should be of different person but should be little bit similar to A it make it hard to learn, which improves model learning. 
**Triplet Loss** - Loss function to learn on triplets of image A,P,N. Later used in gradient descend to learn the NN model.
![](/assets/2019-06-12-2020-06-12-Machine-Learning-Neural-Network61.JPG)

### Face verification and Binary classification
Above approach was good but we can have alternative as, above model is tweaked little bit and used to identify person by comparing the saved feature vectors(calculated and save in DB) against new image feature vector to identify if new user is from database or not. 
![](/assets/2019-06-12-2020-06-12-Machine-Learning-Neural-Network62.JPG)

### Neural Style Transfer. 
Use style of another image(S) to subject image(C) to create new image G. 
![](/assets/2019-06-12-2020-06-12-Machine-Learning-Neural-Network63.JPG)
![](/assets/2019-06-12-2020-06-12-Machine-Learning-Neural-Network64.JPG)

## Sequence Model
It is used when you have sequence of data, such as language translation, NLP, Music creation etc, Video filtration. Here the output can word to word match(Tx = Ty, length of input equal length of output) or even sometime it can't be equal also. Moreover, to predict next block output you might need input data of previous block as well(such NLP sentence meaning is formed when you consider previous word also). Here we use RNN.

### Recurring Neural Network
Previous block A is fed to next block NN calculation.
![](/assets/2019-06-12-2020-06-12-Machine-Learning-Neural-Network65.JPG)

Here, W<sub>aa</sub> and W<sub>ax</sub> is written side by side in one matrix and marked as W<sub>a</sub>, same with a and x. In backprop you try to reduce sum of loss at each output.
![](/assets/2019-06-12-2020-06-12-Machine-Learning-Neural-Network66.JPG)

**Types on RNN**
![](/assets/2019-06-12-2020-06-12-Machine-Learning-Neural-Network67.JPG)

**Language Modeling and sequence generation**
When model is trained on some test sentences, it start to make prediction on next words based on previous words.
![](/assets/2019-06-12-2020-06-12-Machine-Learning-Neural-Network68.JPG)
* In character level model the charters are passed in place of words and vocab is formed from character. This model doesn't work better then word model also it take more computation time and power.

* Vanishing gradient problem, In deep NN initial layer impact is very less on last layer.
* Exploding gradient, In calculation if weights starts increasing exponentially we use gradient clipping to limit the exploding weights.

To remember old words and overcome vanishing gradient descend, we have following options.
* GRU
* LSTM

### GRU - Gated recurrent Unit
Add another weight W<sub>u</sub> with existing W<sub>a</sub>(represented here as W<sub>c</sub>)
![](/assets/2019-06-12-2020-06-12-Machine-Learning-Neural-Network69.JPG)

### LSTM - Better then GRU
It has few more gates to learn. Update, forget and output gates.
![](/assets/2019-06-12-2020-06-12-Machine-Learning-Neural-Network70.JPG)

### BRNN - Bidirectional RNN
Once model learning is complete, BRNN requires full sentence to predict the words between with it's probabibility. It is not much used as it requires full sentence, there much more complex models in place to which work on realtime data.
![](/assets/2019-06-12-2020-06-12-Machine-Learning-Neural-Network71.JPG)
### Deep RNN.
RNN with multiple RNN layer is deep RNN. 

### Word embedding, 
We use feature matrix in place of word vector, so the dimension is reduced from 10k to 300 in our example, if 10k were words and 300 are feature in vector. Feature vector helps us to relate two different words by analogy like gender, royalty and it can derive similarity as man, women then king will have queen , country capital etc.
![](/assets/2019-06-12-2020-06-12-Machine-Learning-Neural-Network72.JPG)


### Cosine similarity vs Squared difference. 
Cosine between two vector tells about how similar they are, big value is they are more similar, cos(0)=1 and small value they are not at all similar cos(90)=1, while squared difference tell different they are, big value they are different small value they are similar. Check the image above. 
![](/assets/2019-06-12-2020-06-12-Machine-Learning-Neural-Network73.JPG)

### Derive Word embeding. 
THere is matrices E, formed formed of 300 feature for 10k word. If you multiple one hot vector of word to E you will get e (word embedding), also you can stack 10k e(word embedding) vector to form E.
![](/assets/2019-06-12-2020-06-12-Machine-Learning-Neural-Network74.JPG)

### Finding target based on Context word.
**Skip Gram Model**
Context and target word are picked randomly from sentence.
Here we have E(Combined embed vector) and &theta;(Softmax) to learn
![](/assets/2019-06-12-2020-06-12-Machine-Learning-Neural-Network75.JPG)

### Negative sampling
Rather then iterating over 10k words to predict using softmax and update 10K e (E) and &theta;, lets take only k sample for ex 5 negative and 1 positive and update E based on only these training set. How to select negative sample is, either pick randomly (issue common word will show up more) or pick using word's frequency weight. P(w) and f(w).
![](/assets/2019-06-12-2020-06-12-Machine-Learning-Neural-Network76.JPG)

### Sentiment Classification.
Tune E which gets e(feature) value and average it up to find out sentiment of sentence, but sometimes one negative word in starting changes the complete meaning of sentence, in this case average doesn't work. RNN is good model to remember outcomes from previous words as well.
![](/assets/2019-06-12-2020-06-12-Machine-Learning-Neural-Network77.JPG)

### Biased learning issue.
Sometime due to more data, such as babysitter are female, nurses are female creates biased learning for NN. But this is not actually the case in real world man can be female and nurses to remove these biases, we pass learned NN to neutralization algorithm which neutralize feature vector for nurse and baby sitter in gender direction. 
![](/assets/2019-06-12-2020-06-12-Machine-Learning-Neural-Network78.JPG)
![](/assets/2019-06-12-2020-06-12-Machine-Learning-Neural-Network86.JPG)

## Sequnece to Sequence generation
When we have stream and we want to generate stream, such as machine translation, image captioning etc.
![](/assets/2019-06-12-2020-06-12-Machine-Learning-Neural-Network79.JPG)

### Conditional language model.
Language generation model(generate sequence of word based on one input) look similar to sequence to sequence generation model. But S2S model give a feature vector output of given input and passed to language generation model which looks more conditional. We have two option at each word generation pick highest probability output word (greedy) or average out all generated word probability(more optimal).
![](/assets/2019-06-12-2020-06-12-Machine-Learning-Neural-Network80.JPG)

### Beam Search
Here we store top B probability words at each output, and based on conditional probability we keep making prediction of next words, in case of B=3, first word will have 3 possible outcome among 10k, next will have 3*10K and only three will be selected and rest discarded. At reaching EOF, the sentence having max probability will be the result. 
![](/assets/2019-06-12-2020-06-12-Machine-Learning-Neural-Network81.JPG)

### Refinement to Beam Search
Normalization- Since conditional probabilities can be very small and cause undeflow errors, in place of product of probability we use sum of log probability, sum of log probability can be big then we normalize by dividing with count of vocabulary words we have or some percentage of that &alpha;
![](/assets/2019-06-12-2020-06-12-Machine-Learning-Neural-Network82.JPG)

### Error analysis
Who is at fault RNN or Beam window, you can find this out by comparing probability of p(y*) vs p($$\hat y$$). 
![](/assets/2019-06-12-2020-06-12-Machine-Learning-Neural-Network83.JPG)

### Attention Model.
**Bleu Score**- Bilingual evaluation understudy. 1 value to validate good translation is working. In this we compare the machine generated sentence with human given reference sentence and see how same they by checking the common word occupance. 
If the sentence is too large Blue score will drop, to maintain it we want to keep translating rather than waiting for EOF. Just like human we read some word and translate, it is call giving attention to few nearby words.
![](/assets/2019-06-12-2020-06-12-Machine-Learning-Neural-Network84.JPG)

Spectrogram
### Speech recognition & Trigger word Detection.
For both we use the same above attention model, but rather than having text as input we have speech which is frequency sliced basis of time, let's say 10 second audio sliced to 10k times and fed in machine to print the sentence. 
Trigger word are trained to give 1 only when the trigger word occurred.
![](/assets/2019-06-12-2020-06-12-Machine-Learning-Neural-Network85.JPG)