---
layout: post
title: Machine Learning with Tensor Flow.
date: 2018-12-13 18:57:06.000000000 -05:00
type: post
parent_id: '0'
published: true
password: ''
status: publish
categories: [Machine Learning Notes]
tags: [Machine Learning]
randomImage: '37'
meta:
  _edit_last: '1'
author:
  login: sanjaypatel2525
  email: sanjaypatel2525@gmail.com
  display_name: sanjaypatel2525
  first_name: ''
  last_name: ''
---
TensorFlow API is built by google and mainly deals with neural network but support other APIs as well such as scikit lear to one liner ML models.
![](/assets/2019-08-21-Machine-Learning-TensorFlow1.png)
### TF estimator
It is library which hold all high level ML algorithm for ex.

**TF Logging** - It has four logging options,DEBUG,INFO ,WARNING, ERROR. tf.logging.set_verbosity(tf.logging.ERROR)

**Tensors** : One number, vector, matrix etc are are form of tensor. Increase in dimension results are stored in tensors. How would you store 10 matrix. You will think array of matrix, and that variable is called tensor.

Tensor flow formally gives two packages, one to deal with Graph and another to run computation on graph (sessions).

**Computational Graph** - Graph is same old neural network graph each node in graph has some purpose(variable,computation,placeholder), input and output. It's a data strucutre well defined in TF. 

Advantage of using graph- portability (Export and share), Easy to understand, parallelism,  

Following are the component of Computational Graph.
Variable, Placeholders, contants, operation, graph, session.

Compared to numpy array they don't allocate memory in starting. For ex.  a = tf.zeros(int(le12),int(le12)) will just form a shape not the actual allocation of memory. It will only allocate memory when it is executed. 

{% highlight PYTHON %}
  import tensorflow as tf
  with tf.session as sess:
    a = tf.constant(12,name="a")
    b = tf.constant(5,name="b")
    prod = tf.multiply(a,b,name="Multiply")
    sum = tf.sum(a,b,name="Sum")
    result = tf.divide(prod,sum,name="Divide")

    out = sess.run(res)
    print(Out)
{% endhighlight  %}

### How the Graph component looks in Code.
Here is very nice explanation from one of the blog.
https://medium.com/@d3lm/understand-tensorflow-by-mimicking-its-api-from-scratch-faa55787170d
{% highlight PYTHON %}
  class Graph():
  def __init__(self):
    self.operations = []
    self.placeholders = []
    self.variables = []
    self.constants = []

  def as_default(self):
    global _default_graph
    _default_graph = self

  ......

  class Operation():
  def __init__(self, input_nodes=None):
    self.input_nodes = input_nodes
    self.output = None
    
    # Append operation to the list of operations of the default graph
    _default_graph.operations.append(self)

  def forward(self):
    pass

  def backward(self):
    pass

  ......

  class add(BinaryOperation):
  """
  Computes a + b, element-wise
  """
  def forward(self, a, b):
    return a + b

  def backward(self, upstream_grad):
    raise NotImplementedError
  # and so on for other operators.

  class Placeholder():
  def __init__(self):
    self.value = None
    _default_graph.placeholders.append(self)

  .......
  # constant can be inputs/labels as they do not changes
  class Constant():
  def __init__(self, value=None):
    self.__value = value
    _default_graph.constants.append(self)

  @property
  def value(self):
    return self.__value

  @value.setter
  def value(self, value):
    raise ValueError("Cannot reassign value.")

  .......
  # Varibale changes in run of graph
  class Variable():
  def __init__(self, initial_value=None):
    self.value = initial_value
    _default_graph.variables.append(self)
  ............

  # Visit and compute graph on topolgical order DFS.
  def topology_sort(operation):
    ordering = []
    visited_nodes = set()

    def recursive_helper(node):
      if isinstance(node, Operation):
        for input_node in node.input_nodes:
          if input_node not in visited_nodes:
            recursive_helper(input_node)

      visited_nodes.add(node)
      ordering.append(node)

    # start recursive depth-first search
    recursive_helper(operation)

    return ordering

  .................
  class Session():
  def run(self, operation, feed_dict={}):
    nodes_sorted = topology_sort(operation)

    for node in nodes_sorted:
      if type(node) == Placeholder:
        node.output = feed_dict[node]
      elif type(node) == Variable or type(node) == Constant:
        node.output = node.value
      else:
        inputs = [node.output for node in node.input_nodes]
        node.output = node.forward(*inputs)

    return operation.output
{% endhighlight  %}

### Important Function used in Andrews NG Course.
* tf.nn.conv2d(X,W, strides = [1,s,s,1], padding = 'SAME') -- convolution layer
* tf.nn.max_pool(A, ksize = [1,f,f,1], strides = [1,s,s,1], padding = 'SAME') - max pool layer
* tf.nn.relu(Z) - element wise relu
* tf.contrib.layers.flatten(P) -- flattens the multidimensional into desire dimension 
* tf.contrib.layers.fully_connected(F, num_outputs) -- Create fully connected layer from F to num_outputs
* tf.nn.softmax_cross_entropy_with_logits(logits = Z, labels = Y) --Calculate cost.
* tf.reduce_mean -- Calculate mean of all example cost.
* Valid vs same pad, same brings add extra padding in case if needed, valid doesn't consider edge to apply filter. (https://stackoverflow.com/questions/37674306/what-is-the-difference-between-same-and-valid-padding-in-tf-nn-max-pool-of-t)
* 