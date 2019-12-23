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
TensorFlow API is built by google and mainly deals with neural network but support other APIs as well such as scikit learn to one liner ML models.
![](/assets/2019-08-21-Machine-Learning-TensorFlow1.png){: .lazyload}

### TF 1 vs 2
* Session and placeholder is removed by tf.function

### TF estimator
It is library which hold all high level ML algorithm for ex. You can create your own estimator by extending **tf.estimator.Estimator** or just use predefined estimator, such as 
**tf.estimator.DNNClassifier** for deep models that perform multi-class classification.
**tf.estimator.DNNLinearCombinedClassifier** for wide & deep models.
**tf.estimator.LinearClassifier** for classifiers based on linear models.
{% highlight PYTHON %}
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,# Two hidden layers of 30 and 10 nodes respectively.
    hidden_units=[30, 10],# The model must choose between 3 classes.
    n_classes=3)
classifier.train(input_fn=lambda: input_fn(train, train_y, training=True),
    steps=5000)
eval_result = classifier.evaluate(
    input_fn=lambda: input_fn(test, test_y, training=False))
predictions = classifier.predict(
    input_fn=lambda: input_fn(predict_x))
{% endhighlight  %}

### TF Keras
Keras is a high level set of estimators, ready to use. 
It requires  
* Model - tf.keras.Sequential()
* Type of layers, layers.Dense(64, activation='relu'), it takes
  * activation
  * kernel_regularizer, tf.keras.regularizers.l1(0.01)
  * bias_regularizer
* tf.keras.Model.compile , it takes
  * optimizer
  * loss
  * metrics
* Fit data - model.fit()
  * epochs
  * batch_size
  * validation_data
* Evaluate - model.evaluate(dataset)
* Predict - model.predict(data, batch_size=32)
{% highlight PYTHON %}
model = tf.keras.Sequential()
model.add(layers.Dense(64, activation='relu'))
or
model.add(layers.Dense(64, activation=tf.keras.activations.relu))
model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=[tf.keras.metrics.Accuracy])

### Fit and run validation
import numpy as np
data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))
val_data = np.random.random((100, 32))
val_labels = np.random.random((100, 10))
model.fit(data, labels, epochs=10, batch_size=32,
          validation_data=(val_data, val_labels))

### or just batch up before putting in fit.
dataset = tf.data.Dataset.from_tensor_slices((data, labels))
dataset = dataset.batch(32)
model.fit(dataset, epochs=10) 

### Evaluate and predict
# With Numpy arrays
data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))
model.evaluate(data, labels, batch_size=32)
# With a Dataset
dataset = tf.data.Dataset.from_tensor_slices((data, labels))
dataset = dataset.batch(32)
model.evaluate(dataset)

{% endhighlight %}

### Declarative approach, take tensor, return tensor. 
{% highlight PYTHON %}
inputs = tf.keras.Input(shape=(32,))  # Returns an input placeholder
# A layer instance is callable on a tensor, and returns a tensor.
x = layers.Dense(64, activation='relu')(inputs)
x = layers.Dense(64, activation='relu')(x)
predictions = layers.Dense(10, activation='softmax')(x)
model = tf.keras.Model(inputs=inputs, outputs=predictions)
# The compile step specifies the training configuration.
model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# Trains for 5 epochs
model.fit(data, labels, batch_size=32, epochs=5)
{% endhighlight  %}


### Callbacks in running
tf.keras.callbacks.ModelCheckpoint: Save checkpoints of your model at regular intervals.
tf.keras.callbacks.LearningRateScheduler: Dynamically change the learning rate.
tf.keras.callbacks.EarlyStopping: Interrupt training when validation performance has stopped improving.
tf.keras.callbacks.TensorBoard: Monitor the model's behavior using TensorBoard.
{% highlight PYTHON %}
callbacks = [
  # Interrupt training if `val_loss` stops improving for over 2 epochs
  tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
  # Write TensorBoard logs to `./logs` directory
  tf.keras.callbacks.TensorBoard(log_dir='./logs')
]
model.fit(data, labels, batch_size=32, epochs=5, callbacks=callbacks,
          validation_data=(val_data, val_labels))
{% endhighlight  %}

### Save & Reload
{% highlight PYTHON %}

# Recreate the exact same model, including weights and optimizer.
model.save('my_model.h5')
model = tf.keras.models.load_model('my_model.h5')

# Save only model to json or yaml
json_string = model.to_json()
fresh_model = tf.keras.models.model_from_json(json_string)
yaml_string = model.to_yaml()
fresh_model = tf.keras.models.model_from_yaml(yaml_string)

# Save weights 
model.save_weights('./weights/my_model')
model.load_weights('./weights/my_model')

# Distributed startegy
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
  model.compile(loss='binary_crossentropy', optimizer=optimizer)
model.summary()


{% endhighlight  %}


### Data preprocess.
{% highlight PYTHON %}

{% endhighlight  %}

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


### Learning on MNIST
{% highlight PYTHON %}
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

{% endhighlight  %}