---
layout: post
title: Machine Learning Linear Algebra
date: 2018-12-15 18:57:06.000000000 -05:00
type: post
parent_id: '0'
published: true
password: ''
status: publish
categories: [Machine Learning Notes]
tags: [Machine Learning]
randomImage: '36'
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
# Linear Algebra
## Vector
Vector is a list, but each item in list defines a different axis. For ex in 2D plain, [x,y] there are two axis x and y. Talking about living being, eyes, nose, ....legs . There are many attribute/dimension. human has 2 legs but cow has 4 in leg axis.
**Vector has magnitude and direction** - If we take only 1 vector a time, i. We can define magnitude by number. But the direction is always relative so the direction will only be shown when there is another vector. Such 3i+4j. 3 x and 4y, x and y are different direction and forms a plain. so 3i+4y gives resulting vector of 3i and 4j vectors. 
### Basis Vector
Axis is called basis vector for ex. 3i+4j is give points in space x to 3 and y to 4 and i,j are axis or basis vector.
### Linearly dependent and independent. 
3i+4j is Linearly dependent to 6i-9j, they both lie same dimension so linearly dependent. 3i+4j ne to 7w. Here w is different dimension.

### Matrix
AX = Y it tells matrix A is combination of rotation and then sheer stacked in matrix A column which transform X to Y. 

### Determinant.
Give the factor by which area has changed after transformation. AX=Y. |A| gives factor by which area change in X to Y coordinate system.

### Inverse
AX=Y, How much inverse of A you should apply to get X. It can be negative as well. But what is |A| is 0. This mean, A has linearly dependent columns which reduces the rank of matrix, for example 3d to 2d or 1d, or 2d to 1d etc. After transforming 3d to 2d or 1d it is difficult reshape it original shape. 

### Dot product
x.y is x projection over y, which x.ycos&theta;, magnitude of x projection over y multiplied by y magnitude.
a · b = ax X bx + ay X by + az X bz