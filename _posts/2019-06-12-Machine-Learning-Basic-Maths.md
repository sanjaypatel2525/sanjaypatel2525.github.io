---
layout: post
title: Machine Learning Basic Maths
date: 2018-12-15 18:57:06.000000000 -05:00
type: post
parent_id: '0'
published: true
password: ''
status: publish
categories: [Machine Learning Notes]
tags: [Machine Learning]
randomImage: '29'
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
This blog covers basic mathematics required for Machine learning. There are many terms which we need to know before we can jump into Machine learning.

#Statistics in Mathematics
**Data & Frequency**: **Data** be anything given as input, for ex, let's say set of alphabets, **frequency** is defined as how many time each alphabets has appeared in the set. Percentage is defined as {% raw %}$$\frac{frequency}{total\,items\,in\,set}$${% endraw %}

**Mean or Average or Population Mean** - Is defined as sum of all elements divided by count of the elements. 
{% raw %}$$\bar x = \frac{\sum X_i}{n}$${% endraw %}

**Sample Mean** - Mean of the sample from whole population. It is represented by $$\\\bar x$$, while population mean is represented by $$\\\mu $$.  It is used for symmetric data set.
**Median** - Put all the number in ascending order and the element at half distance is Median. Used for skewed dataset, doesn't matter if few value are outlier.   
For ex:  Median or M or $$\\\widetilde x$$ of 1,3,4,5,6,7,100,200 would be 5.5. Either middle or mean of middles.

**Mode** - Is most frequent number. For example 1,1,1,2,100,101,103,100 is 1. As the 1 has occurred maximum number of times.

**Quartile** - The set is put in ascending order and divided into 4 quarter of equal width. For ex.
1,3,4,24,29,101,103  has first quartile as 3.5, second quartile/Median as 24 and third quartile as 115.

**Variance** - Measure of being different then other. THere are three variance 1)Population variance 2) Sample variance 3)Alternate formula to calculate variance

**Population variance or just variance** - Is formulated as average  of the square distance from mean. {% raw %}$$\sigma ^2 = \frac{\sum _{i=1}^{N}(x_i-\mu)^2}{N}$${% endraw %}. 

**Standard deviation** - Root of population variance is standard deviation. It gives normalized result against dataset. {% raw %}$$\sigma = \sqrt\frac{\sum _{i=1}^{N}(x_i-\mu)^2}{N}$${% endraw %}

### Permutation & Combination
Permutation is the count of number of ways which one particular Set can be arranged. The count should **consider the order**. For ex. S = {1,2,3} ca be arranged as {321,312,231,213,132,123}. So total number can be arranged as $$3\times2\times 1\\n\times (n-1)\times(n-2)...\\n!$$

But what if number can be repeated.  
$$3\times3\times 3\\n\times n\times n...\\n^n$$

What if the set is {1,2,3,4}, It will be $$4!$$ and what if we need to pick only 2 out 4. n=4, r=2, n-r=2. The possible combination would be,  
$$4\times3$$ = $$4\times3\times\frac{2\times1}{2\times1}$$ 
$$n\times(n-1)\times...(n-r)$$ = $$n\times(n-1)\times...\frac{(n-r)\times(n-r-1)\times...1}{(n-r)\times(n-r-1)\times...1}$$ 
$$P_r^n = \frac{n!}{(n-r!)}\\
\mathbf{If\,p_1,p_2..p_n\,are\,of\,same\,type}\\
P_r^n = \frac{n!}{(n-r!)p_1!p_2!..p_n!}\$$ 

For example, Flip coin 6 times, getting exactly 1 head {HTTTTT,THTTTT,TTHTTT,TTTHTT,TTTTHT,TTTTTH}, Order matters here so permutation hence 6!, but 5T are same so divide by 5!. In case of two heads, Group of two head and group of 4 tail hence, 6!/(4!2!) = 15.

**Combination** is related to permutation, In this the order of number layed out is not important. For ex (1,2) and (2,1) are same. That mean combination is all the permutation deviled by permutation of r. In all the permutation we need to exclude sets which has same set of numbers in any order, let say Set of r can have r! possible permutation and there number of sets so for each set like (12,21) = 2!  = 2, one has to be excluded. Let' say Set {1,2,3} and need to pick two number.  
$$\require{cancel} P_2^3 = \frac{3!}{(3-2)!}= 6, \{32,31,23,21,13,12\}\\
C_2^3 = \frac{3!}{(3-2)!(2)!}= 3, \{32,31,\cancel{23},21,\cancel{13},\cancel{12}\}\\
C_r^n = \frac{n!}{(n-r!)r!} = \frac{P_r^n}{r!}\\
C_r^n\subseteq P_r^n\subseteq n! \subset n^n$$

In case there are multiple groups of same type g<sub>1</sub>,g<sub>2</sub>..g<sub>n</sub> and every we need r<sub>1</sub>,r<sub>2</sub>..r<sub>n</sub> number from each group respectively. The combination would be $$C_{r_1}^{g_1}\times C_{r_2}^{g_2}\times..C_{r_n}^{g_n},$$  
For example a comity 3 people of 1 man and 2 women from a group 2 men and 3 women has to formed. M1,M2 and W1,W2,W3. Picking them in any order doesn't matter so here commutation has to be calculate. 2C1 and 3C2 = 6.

### Probability
Probability is a measure of how likely an event is going to happen from given set of event. For example. {1,2,3,4,6} are equally likely outcome of a dice. Then probability of coming to in a dice roll would be 1 out of 6. Probability of dice showing number 2.
$$P(2) = \frac{1}{6},\\ P(x) = \frac{count\,of(event\,x)}{total\,count\,of\,all\,event}\\P(x+y) = \frac{count\,of\,event\,x+y}{count\,of\,all\,events}$$.

**Probability in relation to probability** Lets say A is event of 1 and 4 coming on dice and B is event {1,3,4} coming on dice. Intersect of A and B is {1,4}
$$P(A) = \frac{2}{6},\;P(B) = \frac{3}{6};\\
P(AUB) = P(A)+P(B)-P(A\cap B )  = \frac{1}{3}+\frac{1}{2}-\frac{2}{6} = \frac{1}{2}$$  

If A and be are **independent ** A is dice and B is deck getting 1 on dice and an spades ace on dec is
$$P(A\cap B) = P(A).P(B) = \frac{1}{6}\cdot\frac{1}{52} = \frac{1}{312}$$ 

**Conditional probability on Dependent event** Dependent event are the one which affect the event space for next event. For example in a deck probability of getting Ace is 1/13 and then you put that Ace aside the card deck size is now 51, now the probability of getting ace is 3/51. Here A has affected the event space for B. So we can write P(A) then P(B) is,  
{% raw %}$$P(A\,and\,B) or P(A\cap B) = P(A).P(B|A)\ = P(B).P(A|B)\;or\\
P(B|A) = \frac {P(A\,and\,B)}{P(A)}$$  {% endraw %}

Where P(B\|A) can be defined as P(B) if P(A) has already occurred. P(B and A) are calculate over two dice throw, while P(B\|A) is probability of second dice throw. Extend to third variable.  
{% raw %}$$P(A|B,C)=\frac{P(A\cap B\cap C)}{P(B\cap C)}$${% endraw %}

**Random Variable** - Random variable is possible outcome of any event. Let's say in a particular match how many goals will be made, it can be 1 or 4 or 6 or any number. This is discrete and can be counted and has some interval. Set containing 0 to 100 is discrete, but different floating values between is infinite/uncountable. Thing which can't be counted are **continuos random variable** and which can be counted are discrete. Continuos value can be measure but not counted for example volume.

**Probability distribution** - Do not relate this with single probability ie (probability of getting 4 in dice.). Think it as sheet where first column is possible outcomes and second column is probability of that outcome. In this case the probability of each possible outcome is different. Fox ex. Let there are some event E1, E2 ..En from Sample Space S. And let X be function over combination of probabilities over multiple E. For ex. Flip coin three times and X is count H in this sample space. X will variate from 0 to 3.
X=0 {TTT} - 1, X=1 {HTT, THT, TTH} - 3, X=2 {HHT,HTH,THH} -3, X=3 {HHH} - 1.
$$ X\,be\,{x_1,x_2,x..}\; then\\P(x_1)+P(x_2)+P(x_3)+... = 1 = \sum_{i=0}^{n}p(x_i)$$

**Probability Mass function** - Mass function comes into picture when the values are discrete and probability weight calculate by sum, in contrast to PDF which is integral.
$$pX(x) ≥ 0\;and\;\sum_x pX(x) = 1$$

**Probability Density Function** - In a set of continuos event probability normally measure between two points and the area under two defines the probability density function. The entire area is calculated as one. PDF exist only for continuos variable and the area under two interval is the probability. Let's say x is random variable and f(x) is the probability of the x variable then.
{% raw %}$$
P(a\leq x \leq b) = \int_a^b f(x)dx \\
\int_{-\infty}^{+\infty} f(x)dx = 1
$${% endraw %}
* Probability of normally distribute function looks like bell shaped and has fixed are under two interval.
* Continuos functions are calculated by integral while discrete with summation.
![](/assets/2019-06-12-Machine-Learning-Basic-Maths5.JPG){: .lazyload}

**Probability Commutative function** - Probability of being equal to or less then x, F(x)=P(X≤x)

**Random Variable X - &mu;** - Average of probability random variable ie function over sample space. 
**Mean or Expected value over discrete function** 
{% raw %}
$$E(X)=\mu=\sum_{i-1}^nx_ip(x_i)\\
\mathbf{Continuos \, variable}\\
E(X)=\mu=\int_{-\infty}^{+\infty}x_if(x)dx\\
\mathbf{Lotus-Law\,of\,Unconscious\,Statistician}\\
E(g(x))=\sum_{\infty}g(x)P(X=x)\\
E(g(x))=\int_{-\infty}^{+\infty}g(x)f(x)dx\\
\mathbf{Variance\,of\,random\,variable}\\
Var(X)=\sigma^2=E{(X-\mu)}^2=\sum_{i-1}^n{(x_i-\mu)^2}p(x_i) = E((X-E(X))^2) = E(X^2)-(E(X))^2\\\\
\mathbf{Standard\,deviation}
SD(X)=\sqrt{Var(X)}=\sigma=\sqrt{\sum_{i-1}^n{(x_i-\mu)^2}p(x_i)}=\sqrt{\sum_{i-1}^nx_i^2p(x_i)}\\
$$
{% endraw %}
**Standard deviation** - How far the numbers are from mean on average, think of population having lot of small numbers and then some big numbers so mean doesn't in middle but at the one end. In this case mean and standard deviation will be far.
**Covariance** - Measures tendency of x and y deviate from their mean in same or opposite direction at same time.
$$cov(x,y)=  \sum_ip(x_i,y_i)(x_i-\mu_x)(y_i-\mu_x)\\
\mathbf compare\,to\,actual\,covariance(not\,in\,probability\,space)\\
\sum_ip(x_i)=\frac{1}{N-1} \sum_1^n \\
cov(x,y)=  \frac{1}{N-1}\sum_1^n(x_i-\mu_x)(y_i-\mu_x)
$$
![](/assets/2019-06-12-Machine-Learning-Basic-Maths6.JPG){: .lazyload}
**Correlation** - pearson's correlation coefficient is covariance normalized by standard deviations of variables.
$$corr(x,y)=\frac{cov(x,y)}{\sigma_x\sigma_y}$$ 

**Mean or Expected value over continuos function**
$$E(X) = \int_{x=a\to b}xp(x)$$

**Bayes Theorem** - Let A<sub>1</sub>, A<sub>2</sub>,A<sub>3</sub>..A<sub>n</sub> are sample set out of S where all including form S. Let B is another set from S, since it is part of S it definitely form out of parts of A's. Hence,  
B = (B &#8745; A<sub>1</sub>)U(B &#8745; A<sub>1</sub>)..U(B &#8745;A<sub>n</sub>) and we can write.
$$P(A_k|B) = \frac{P(A_k\cap B)}{[P(A_1\cap B)+P(A_2\cap B)+..P(A_n\cap B)]}\\
Using\;P( A_k \cap B ) = P( A_k )P( B | A_k )\\
P( A_k | B ) =  	\frac{P( A_k ) P( B | A_l )}{[ P( A_1 ) P( B | A_1 ) + P( A_2 ) P( B | A_2 ) + . . . + P( A_n ) P( B | A_n ) ]}$$ 
![](/assets/2019-06-12-Machine-Learning-Basic-Maths7.JPG){: .lazyload}

### Discrete Distribution
**Bernoulli Trials** - Trials which answers only in two values, such yes/no, 0/1, head/tail etc. Such that $$p=\frac12\;and\; q=1-p=\frac12$$.

**Bernoulli Distribution** - A sheet with with column1 as possible outcomes as 0 and 1 and coloumn2 as there probability. For ex, p = 0.15 and q= 0.85.
![](/assets/2019-06-12-Machine-Learning-Basic-Maths9.JPG){: .lazyload}
$$E(X) = 0.q - 1.p = p\\
V(X) = E(p^2)-E(p)^2 = p-p^2$$.


**Binomial Distribution** - Here we talk about Bernoulli sample space. Let say function variable X is times we get Head and call it S and other is F. Flip 3 times (SSS,SST,STS,STT,TSS,TST,TTS,TTT). p is success probability and q is failure.  

| X | 0 | 1 | 2 | 3 |
| --- | --- | --- | --- | --- |
| P|q<sup>3</sup> | 3q<sup>2</sup>p | 3p<sup>2</sup>q | p<sup>3</sup> | 

$$P(S)=q^3+3q^2p+3qp^2+p^3=1\\
\sum_{i=0}^nP(x_i) = \sum_{i=0}^nC_x^nq^{n-x}p^x=1\\
\mu=np\;,Var(X)=npq$$
![](/assets/2019-06-12-Machine-Learning-Basic-Maths11.JPG){: .lazyload}

**Hypergeometric Distribution** - Binomial but sample space reduced due to last event, for ex. draw colored ball from bag starting with equal probability, but do not place the ball back which reduces the sample space.  
![](/assets/2019-06-12-Machine-Learning-Basic-Maths12.JPG){: .lazyload}

**Uniform distribution** - Fair dice role has 6 possible outcome and each has probability of 1/6. Since 1 is total probability, area = width*height = f(x)(b-a) = 1; f(x) = 1/(b-a)
![](/assets/2019-06-12-Machine-Learning-Basic-Maths13.JPG){: .lazyload}
![](/assets/2019-06-12-Machine-Learning-Basic-Maths10.JPG){: .lazyload}
$$\mu= \int_a^bxf(x)dx = \int_a^bx\frac{1}{b-a}dx = \frac{a+b}{2}\\
Var(X)=\frac{(b-a)^2}{12} $$

**Poisson Distribution** - Type of binomial but rate defined by &lambda;, The distribution when you know the constant rate of event over time or space. For ex. 50 email per hours or 22 trees per kilometer etc. Second thing the event are independent. For ex, 20 call per minute, then 0.33 call per second. Every second either call can come (0.33) or not(0.66). Let &lambda;	be probability of call per minute, divide that in n=60 interval then probability of on slice of n is p=&lambda;/n and q=1-&lambda;/n.
![](/assets/2019-06-12-Machine-Learning-Basic-Maths14.JPG){: .lazyload}
$$P(X=x) = C_x^np^xq^{n-x} =  \frac{n!}{(n-x)!x!}{\frac{\lambda}{n}}^x\left({1-\frac{\lambda}{x}}^{n-x}\right)=\frac{\lambda^xe^{-\lambda}}{x!} \\
E(X)=\mu=\lambda,\;Var(X)=\mu=\lambda$$ 

**Negative Binomial Distribution** - Number of failure before you get specific number of success. In contrast to Binomial, how many success after x trials. this is how many failures needed to get k successes and this gives how many trials. let k be number of successes in x trial then x-k is number of failure.  

**Geometric Distribution** - Type of negative Binomial where you are interested to get first success after r failures. Number of trial becomes x = r+1. We are looking at kth trial which is success.
![](/assets/2019-06-12-Machine-Learning-Basic-Maths15.JPG){: .lazyload}
$$P(X=k) = (1-p)^{k-1}p\\
E(X)=\mu = 1/p,\;P(Failure) = \frac{(1-p)}{p}\\
Var(X) = \frac{1-p}{p^2}$$

### Continuos Distribution

**Normal Distribution** - Distribution where mean, mode and median coincide. It is bell shaped, there are equal exactly half of the value on left and right side. 
$$f(x) = \frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}\\
E(X)=\mu,\;Var(X)=\sigma^2$$
![](/assets/2019-06-12-Machine-Learning-Basic-Maths16.JPG){: .lazyload}
Normal distribution is not related to pro

**Central Limit Theorem**- CLT (center, shape and spread), states even if we take an uneven distribution and take sample from it for n times, find the mean of each sample, start putting mean in buckets, the graph formed due to this filling will be more like normal distribution.**Mean of sample means** distribution will depict mean of population.**Standard Error** of population will be standard deviation of distribution sample means.
$$SE = \frac{\sigma}{\sqrt{N}}$$

**Confidence Interval** - In statistics confidence level defined by confident intervals. How confident are wo on our intervals. Let's say population sleeps for 6-10 hours is 95% confidence.  
The confidence level, tells us how confident we are, that this particular interval captures the true population mean. We never deal with full population we go by multiple samples of population and with that we define population mean by calculating distribution of samples means.


**Exponential Distribution** - It is mix of poisson where we have rate, like calls per minute and geometric where we are interested in wait time of next call or how many failure before next success. **Weibull** is counterpart of exponential, time to failure. Ex. Machine failure time when we know rate.
![](/assets/2019-06-12-Machine-Learning-Basic-Maths17.JPG){: .lazyload}
$$F(x) = \lambda e^{-\lambda x}\\
E(X)=\mu = 1/\lambda,\\
Var(X) = \frac{1}{\lambda^2}$$

## Calculus
How function changes over times(**derivatives/by differentiation**), how they accumulate over time period(**integral/ by integration**).

### Derivative
It cabe defined in two ways. 
* In geometry - slope of line at specific point ie, y=mx+b, m becomes slope.
* In physics - Rate changes at instant. 

**In Geometry**  
The slope of line defined by following formula in between any two points. 
$$Slope = m = \frac{y_1-y_2}{x_1-x_2}= \frac{f(x_1)-f(x_2)}{x_1-x_2}$$
![!img](/assets/2019-06-12-Machine-Learning-Basic-Maths1.JPG){: .lazyload} 

Here we talk about two point if two points are very close such that $$x_1-x_2 \to 0$$, here derivatives comes for rescue. The derivative formula.
{% raw %}$$\Delta x= x_1-x_2 =h\\
\frac{d}{dx}f(x)=\lim_{h\to 0}\frac{f(x+h)-f(x)}{h}\\
\mathbf{Ex}\;f(x) = x^2\\
\frac{d}{dx}f(x)=\lim_{h\to0}\frac{{(x+h)}^2-x^2}{h}=\lim_{x\to0}2x+h = 2x$${% endraw %}

Derivatives are used in optimization problems of machine learning. It helps us to determine to increase or decrease weights in order to achieve maximum or minimum output in  gradient decedent.

**Chain rule**: $$\frac{df}{dx}=\frac{dh}{dg}\frac{⋅dg}{dx}\\
\mathbf Ex:\; f(x) = h(g(x))\, where\, g(x)=x^2\,and\,h(x)=x^3\\
f(x) = (x^2)^3, g'(x)=2x, h'(x)=3x^2\\
f'(x)=h'(x).g'(x)= 3(x^2)^2.2x=6x^5$$

The reason we put double square as for h(x) was derived over g(x) where x = x<sup>2</sup>. The same goes with multiple chains.


### Gradient.
Gradient is a variable which hold partial derivative multivariable function. Partial derivate of function is keeping all other variable constant and find derivative over one. 
$$f(x,y,z)=2z^3x^2y^7\\
\nabla f(x,y,x)=\begin{bmatrix}\frac{df}{dx}\\\frac{df}{dy}\\\frac{df}{dz}\end{bmatrix}=\begin{bmatrix}4z^3xy^7\\14z^3x^2y^6\\6z^2x^2y^7\end{bmatrix}
$$

**Direction Derivative** - While trying for minimum maximum by variating a single variable and keeping others constant we can also apply directional derivative in order to check what would happen if take a small nudge to our current slope, what if in place of north go south. 
$$\vec v.\nabla f(x,y,x)=\begin{bmatrix}2\\3\\-1\end{bmatrix}.\begin{bmatrix}\frac{df}{dx}\\\frac{df}{dy}\\\frac{df}{dz}\end{bmatrix}$$

* Gradient always points to the direction of greatest increase or decrease of a function.
* Gradient reaches to zero in case of maxima and minima.

### Integrals
Integrals can be defined as how much accumulated over time, in other terms how much area is covered under a slope between two points.
![](/assets/2019-06-12-Machine-Learning-Basic-Maths2.JPG){: .lazyload}

$$F(x) = \int_a^b f(x)dx \\
Areas(a,b) = F(b)-F(a)= \int_0^b f(x)dx-\int_0^a f(x)dx$$
![](/assets/2019-06-12-Machine-Learning-Basic-Maths3.JPG){: .lazyload}

**Important Formula**  
![](/assets/2019-06-12-Machine-Learning-Basic-Maths4.JPG){: .lazyload}

**Common Usage of Integral**
* Probability under PD - $$\int_{-\infty}^{+\infty} p(x)dx=1$$.
* Expected Value - $$\int_{-\infty}^{+\infty} xp(x)$$.
* Variance - $$\sigma^2=\int_{-\infty}^{+\infty} (x-\mu)^2p(x)$$.

## Linear Algebra
### Vector
Vector is variable stores direction and it's magnitude from center. In 2D, left 7 and up 2 gives direction m = -2/7 and magnitude as well. It can be stored in 1D array/matrix column or raw anything [-7 2]. It is denoted mostly by bold italic latter with arrow on top. $$\vec v = (a_x,a_y)\\$$
![](/assets/2019-06-12-Machine-Learning-Basic-Maths8.JPG){: .lazyload}
$$\mathbf{magnitude} = |v|(not\,absolute)=||v||= \sqrt{x^2+y^2}\\
x=rcos\theta,\; y=rsin\theta\\
r=\sqrt{x^2+y^2},\; \theta = tan^{-1}(y/x)\\ 
\begin{bmatrix}a\\b\end{bmatrix}+\begin{bmatrix}c\\d\end{bmatrix}=\begin{bmatrix}a+c\\b+d\end{bmatrix}\\
\begin{bmatrix}a\\b\end{bmatrix}-\begin{bmatrix}c\\d\end{bmatrix}=\begin{bmatrix}a-c\\b-d\end{bmatrix}\\
\begin{bmatrix}a\\b\end{bmatrix}/\begin{bmatrix}c\\d\end{bmatrix}=\begin{bmatrix}a/c\\b/d\end{bmatrix}\\
\mathbf{Hadamard\,product}\begin{bmatrix}a\\b\end{bmatrix}\odot\begin{bmatrix}c\\d\end{bmatrix}=\begin{bmatrix}a.c\\b.d\end{bmatrix}\\$$ 

Vector need not to be scaler only they can be a function. Let say x,y is the point on the plane and if it is applied to a vector f(x) = x<sup>2</sup>. In this case depending on X vector value will variate. 

**Dot product** - Multiplication
$$A = \begin{bmatrix}a\\b\end{bmatrix}, B = \begin{bmatrix}c&d\end{bmatrix}\\
A.B = ac+bd\\
\vec a.\vec b = |a||b|cos\theta,\; theta\,is\,angle\,between\,\vec a\; \vec b$$

**Projection** - $$\vec b$$ can make projection over $$\vec a$$ it is the base of the triangle formed by a and b. 
$$proj_ab=\frac{\vec a\vec b}{|\vec a|^2}\vec a$$

### Matrix
Matrix is rectangular grid to store different variable or scaler numbers. You can visualize it as 2D array. Addition and subtraction on matrix over scaler values applies to all the elements and if it is added or subtracted by another matrix then each element will be added or subtracted on each element of other matrix. Multiplication follows the same concept as Dot product and division is not straight forward it is calculate by multiplying with inverse of the matrix. Division is only possible if dividend determinant is non-zero and is a square matrix. 
$$\begin{bmatrix}a&b\\c&d\end{bmatrix}\pm1=\begin{bmatrix}a\pm1&b\pm1\\c\pm1&d\pm1\end{bmatrix}\\
\begin{bmatrix}a&b\\c&d\end{bmatrix}\times\div1=\begin{bmatrix}a\times\div1&b\times\div1\\c\times\div1&d\times\div1\end{bmatrix}\\
\begin{bmatrix}a&b\\c&d\end{bmatrix}.\begin{bmatrix}w&x\\y&z\end{bmatrix}=\begin{bmatrix}aw+by&ax+bz\\cw+dy&cz+dz\end{bmatrix}\\
A*A^{-1}=I\\
AB\neq BA\\
A/B=AB^{-1}\neq B^{-1}A\\
(AB)^T=B^TA^T$$
