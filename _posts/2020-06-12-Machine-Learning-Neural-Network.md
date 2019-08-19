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
tags: []
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

**Loss or Cost Function** - Defines accracy of prediction with given neural network model. 

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
##Back propogation
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


Commuatively we can say $$\delta_j^L=\frac{\delta C}{\delta z^l_j}$$. If this value big that mean changes at this z will make big impact to lower down the C rather then other lower derivative.

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
**Cross Entropy Function** - (Not same as probability distribution cross entropy, do not get confused). In place quardratic cost function (y-a)<sup>2</sup> use cross entropy function to calculate change required for weigh and biases. Learning rate doesn't slow down as quadratic cost function.
$$C=−1n∑x[ylna+(1−y)ln(1−a)]$$

**SoftMax and loglikelyhood** - In place of 0 to 1 output probability on each output layer(sigmoid activation), if we say combined output layer probability should be 1(Softmax activation). Then the data is more comparable. Here cost function used is -log a, in case of output close to 1 less change is needed and in 0 more change is required.
 $$\sum_j a_j^L=\frac{\sum_j e^{z_j^L}}{\sum_k e^{z_k^L}}\\
 C=-log\;a_y^L$$

**overfitting or overtraining** - the epoch from where you dont see much learning after that.

**Bias & Variance Tradeoff** - Bias is difference between average predicted value to true value. High bias Pays very little attention to training set and oversimplfies the model(**underfitting**). For ex. Model devices is lenear function while the actula need was non lenear function. Variance tells about the spread of the data. High variance over learn the training set and doesn't generalize it (**overfitting**). It picks the outliers/noise and noise also in its knowledge which is overfiting of the data.  High parameter increase high variance and low parameter increase high bias, we have to trade off in selecting right parameter.
$${\displaystyle \operatorname {E} {\Big [}{\big (}y-{\hat {f}}(x){\big )}^{2}{\Big ]}={\Big (}\operatorname {Bias} {\big [}{\hat {f}}(x){\big ]}{\Big )}^{2}+\operatorname {Var} {\big [}{\hat {f}}(x){\big ]}+\sigma ^{2}}\\
=(f-E[\hat f])^2+Var[y]+Var[\hat y]\\
=(f-E[\hat f])^2+E[\varepsilon^2]+E[(E[\hat f]-\hat f])^2]$$

**Detect underfit and overfit** - Devide training data in training and test data and use test data to get the accuracy of model. 
**Fixing underfit and overfit**
* Cross-validation - Divide data in splits and train and test. For ex K-Fold, divide data in K and keep increasing the training set, keep decreasing the test set. 
* Train with relevant data- More data can be good, but if it noisy it is issue so train with relevant data.
* Remove feature - Remove irrelevant features. Rubber duck debugging.
* Early stop - More training overfits the data sometime so know when to stop.
* Regularization - Make model simpler sometime, prune decision tree, dropout neural netwrok, penalty paramerter.
* Ensembling - Multiple model learn separately and cobmbine them in the end to smooth it out. Bagging and boosting are example. Bagging start with complex moddle then smooths it out while boosting start with weak learner models and form a complex model.  

**Weight decay Regulazrization- L2 or Ridge Regulazrization** - Add a extra term regularization term, $$C=C_0+\frac{\lambda}{2n}\sum_ww^2\\
\frac{\delta C}{\delta w} = \frac{\lambda}{n}w$$. It helped in overcoming overfitting issue and increasing accruracy, also saved us from local minima.


**Weight decay Regulazrization-L1 or Lasso Regulazrization (LAD - Least Absolute deviation)** - Since 

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