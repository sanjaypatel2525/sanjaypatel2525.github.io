---
layout: post
title: Machine Learning Python Libraries
date: 2018-12-14 18:57:06.000000000 -05:00
type: post
parent_id: '0'
published: true
password: ''
status: publish
categories: [Machine Learning Notes]
tags: [Machine Learning]
randomImage: '31'
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

This is not a extensive detailed guide for machine learning libraries. It is a small extract of common feature of these libraries and there usage. 

There are many Python packages used in ML, but here are some important ones.
* NumPy - Everything related to Maths, formulas, series, multidimensional arrays and operations etc.
* SciPy - For statistical analysis, extension of Numpy
* Pandas - Dealing with data reading/writing/manipulation from/to CSV,database,excel etc, cleaning, filtering.
* Seaborn - For data visualization
* Matplotlib - For data visualization, Seaborn is extension over Matplotlib.
* Scikit-learn - Provides ready to use ML algorithms, based on NumPy, SciPy.

# Numpy
## Numpy array
{% highlight Python %}
  import numpy as np
  a = np.array([1, 2, 3])
  # a = np.array([1, 2], dtype=np.int64) define datatype manually
  print(type(a)) # Prints "<class 'numpy.ndarray'>"
  print(a.shape) # Prints "(3,)"
  a[0] = 5 # Change an element of the array
  print(a) # Prints "[5, 2, 3]"
  b = np.array([[1,2,3],[4,5,6]])    # Create a rank 2 array
  print(b.shape) # Prints "(2, 3)"

  # initialize with zeros, ones, any other number by np.full((2,2), 7)
  a = np.zeros((2,2)) # [[ 0.  0.][ 0.  0.]]
  # eye for diagonal filled with given number and others zero.
  a =  np.eye(2)  # [[ 1.  0.] [ 0.  1.]]

  # Slicing
  a[1, :] # second row and all column. It will be 1 D, rank 1
  a[1:2, :] # second row and all column but it will be of 2 rank or 2D.
  # a[[0, 0], [1, 1]] is equivalent to [a[0, 1], [0, 1]]
  # a[[0, 1, 2], [0, 1, 0]] is equivalent to [a[0, 0], a[1, 1], a[2, 0]])

  # Evaluate condition
  bool_idx = (a > 2) # prints true or false based on condition[[False False][ True  True][ True  True]]
  # a[bool_idx] and  a[a > 2] will print 1D array of all value matching the criteria.

  # array maths
  print(x + y)
  print(np.add(x, y))  # same as above, all other operator are like this.

  # matrix multiplication
  np.dot(x, v) or x.dot(v)

  # Generate array
  np.arange(10,20,2) # will start form 10 upto 20 step by 2.

  # reshape array
  np.arange(12).reshape(4,3)

  # Flat, flatten, ravel
  a.flatten(order = 'F')

  # Sum
  a.sum() # sums up all element
  a.sum(axis=(i,j,..))) # sum over i j .. axis


  # get Diagonal
  np.diagonal()
  np.diagonal(offset=1)
  
  # Sum over principal diagonal or trace
  np.trace()

  # using random
  np.random.seed(1)
  
  #Generate 3 random integers b/w 1 and 10
  print(np.random.randint(0,11,3))

  # get random from normal distribution
  print(np.random.normal(1.0,2.0,3))

{% endhighlight  %}

* Numpy gives a class which helps to create homogeneous structure, you can say numpy defined datatypes. Here is signature, need not to remember.
{% ihighlight PYTHON %}
  numpy.dtype(object, align, copy)
  dt = np.dtype([('age',np.int8)]) 
  a = np.array([(10,),(20,),(30,)], dtype = dt) 
  np.sort(a, order = 'age') # can sort by custom defined objects
{% endihighlight  %}

* A closure look to array.
{% ihighlight JAVA %}
numpy.array(object, dtype = None, copy = True, order = None, subok = False, ndmin = 0)
a = np.array([1, 2, 3], ndmin=2,dtype = complex) 
# [ 1.+0.j,  2.+0.j,  3.+0.j]  
{% endihighlight  %}

* Read order of n dimension. C -(2,3,4) the read sequence will variate on 4 most then 3 and then 2. F- (2,3,4) 2 will variate most. It doesn't impact how array store in memory, only read mode is changed.

* Iterate ove array
{% ihighlight PYTHON %}
for x in np.nditer(a, order = 'F',op_flags = ['readwrite']):  # default order is C, OP flags defaults to read.
   print x,
  
# broadcasted iteration of vector and matrix
for x,y in np.nditer([a,b]): # x(3,4) and y is(1,3) or y(3)
   print "%d:%d" % (x,y),
{% endihighlight  %}







## Numpy linear algebra
{% highlight PYTHON %}
  a = np.arange(9).reshape(3,3)
  b = np.arange(9).reshape(3,3) 
  dot(a,b,result) # matrix multiplication
  vdot(a,b) #element vise
  inner(a,b) # u<sup>T</sup>
  outer(a,b) # uv<sup>T</sup>
  pinv(a) # psudo inverse
  inv(a) # true inverse
  np.linalg.eig(a) #eign value
  np.linalg.norm(x[, ord, axis, keepdims]) # 1,2,p,infinity
  np.linalg.matrix_rank(M[, tol, hermitian])
  np.linalg.solve(a,b) # solve for x where ax=b
  np.linalg.tensorsolve(a,b) # solve for x where ax=b, in n dimension


{% endhighlight  %}
**Eign vector and value** of Matrix is arbitrary &lambda; and v for a matrix A. which specifies ratio of vector dimensions and a magnitude respectively.  Eign vector ratio sum need not to be 1 but just to give better picture they are scaled to sum 1.

**Pseduo inverse** - Sometime A<sup>-1</sup> is not defined as |A| is 0. So we calculate X which is very close to A<sup>-1</sup>.
$$A^{-1}=I, AA^{-1}-I = 0, AX-I \text{ is close to 0}$$

**Harmition Matrix** - $$A_{ij} = \bar A_{ij}$$. Conjugate matrix of complex number and transposed.

**Echelon Matrix** - Matrix starts with 1 and every new row start 1 column after.

**Orthogonal Matrix** - AA<sup>T</sup>=I.

**Matrix Rank** - Calculate by reducing the lenearly dependent rows. Minimize it by row to row computaion and try to make them 0. It is computed by bringing matrix to echelon matrix form.

**LU decomposition** - A = LU. L is lower traingular matrix and U is upper triangular matrix.

**Cholesky Decomposition** - A = LL<sup>T</sip>, where L is hermitian matrix.

**QR Decomposition** - A =QR where Q is Orthogonal Matrix and R is upper traingular.

**SVD** - A = USV<sup>T</sup> where U<sup>T</sup>U=I, V<sup>T</sup>V =I


## Pandas
{% highlight Python %}
import numpy as np
import pandas as pd

# Types series and dataframe.
xyz = pd.Series([1, 3, 5, np.nan, 6, 8])
xyz = pd.DataFrame([[100,200,300],[400,500,600]],columns=[],index=[])

# Read Data from file.
xyz = pd.read_csv('xyz.csv', delimiter = ';') # read csv
xlsx = pd.ExcelFile('file.xls') # read all sheets in excel
xyz = pd.read_excel(xlsx,  'Sheet1') # read sheet in excel

# Read from SQL
from sqlalchemy import create_engine
engine = create_engine('sqlite:///:memory:')
pd.read_sql(SELECT * FROM my_table;, engine)
pd.read_sql_table('my_table', engine)
pd.read_sql_query(SELECT * FROM my_table;', engine)
xyz.to_sql('myDf', engine)

# provide customer headers and index name
xyz  = pd.read_csv('xyz.csv', delimiter=';', names = ['firstname', 'lastname', 'country', 'user_id', 'source', 'topic'])

# head, tail, random
xyz.head()
xyz.tail()
xyz.random(5)

# Get the data
xyz.iloc([0], [0]) or xyz.iat([0], [0]) # get row=0,column=0 value
xyz.loc([0],  ['Country']) # by label
xyz[\['firstname','lastname']] # it will return dataframe
xyz['firstname'] or xyz.firstname # will return series


# filter data
xyz['firstname'='abc']
xyz[xyz['col1']>10 & xyz['col2']='san']
# or with some condition ie starts with.
condition =  xyz.firstname.startswith('a', na=False)
xyz = xyz[condition]

# Remove data.
xyz.drop([val1,val2]) // if indexes has name, drop based on indexes.
xyz.drop(xyz.index([2,-3])) // drop row with index 2 and 3 from the last.
xyz.drop([col1,col2],axis=1) // remove columns from dataframe
xyz.drop(xyz[df.col1 < 50].index) //find and drop by index.


#Handle null or na
xyz.dropna() or xyz.dropna(axis=1) @ dropping the row or column who have null values.
xyz.isnull().sum() # get count of null records.
xyz.fillna(meanofxyz, inplace=True)


# Aggregate
xyz.count() # will print count of non na of each column
xyz.sum(), max, min, mean, median
xyz[col1].value_counts() # does group by on col1 and counts the frequency

# custom function commulative over column
somfunction = lambda x: x.max()-x.min()
xyz.applymap(somfunction)

# Series map function -same as applymap
xyz['col1'].map(format)

# custom function applied on each element such change format of data
format = lambda x: '%.2f' % x
xyz.apply(format)

#Group By - Group on some column and aggregate on others
xyz.groupby('animal').mean()

# Joins. By default panda find join key to join by matching name if not you have to specify separately. Also, you can choose join strategy inner, outer,left, right. Let's say zoo has animals and other columns and zoo_eats has animal and what animal eats.
zoo.merge(zoo_eats, how = 'left', left_on = 'animal', right_on = 'animal')

# fill missing
xyz.fillna('unknown')

# Sorting.
xyz.sort_values(col1) or xyz.sort_values(by = ['col1', 'col2'], ascending = False)

#reset index after sort sequence is changed.
xyz.reset_index()
xyz.reset_index(drop = True) # remove old index column.

{% endhighlight  %}

