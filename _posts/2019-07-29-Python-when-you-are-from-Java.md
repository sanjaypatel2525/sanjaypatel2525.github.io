---
layout: post
title: Python when you are from Java
date: 2018-12-14 18:57:06.000000000 -05:00
type: post
parent_id: '0'
published: true
password: ''
status: publish
categories: [Machine Learning Notes]
tags: []
randomImage: '32'
meta:
  _edit_last: '1'
author:
  login: sanjaypatel2525
  email: sanjaypatel2525@gmail.com
  display_name: sanjaypatel2525
  first_name: ''
  last_name: ''
---
Let's see some Python Basics first you can skip it as well. 

* Python string can be inside single quotes and double quotes and three quotes as well. Three quotes takes care of new line as well.
* Either use 4 spaces or a tab to indent.
* \# for comment and """ for multiline documenttaion comment
* Naming conventions
  * Module name, variable name and function name should be all small with underscores.
  * Class name should CaptWord. 
  * Constants should be ALL_CAPS with underscore.

### Type of methods in class
There are three type of methods in class is **instance methods**(default, first paramerter has to be self and gets passed automatically. It requires object to call these methods, Access class with self.__class__), **class methods** (first parameter is cls and class reference gets passed automatically, directly call using classname), **static methods** (nothing is passed and can be accessed by classname).

### Underscore usage.
  * Single prefix underscore - In wildcard import methods/variable with name starting with underscore are not imported.
  * Single suffix underscore - Sometime you want use reseverd keyword but they are already taken so you use suffix underscore.
  * Double prefix underscore - The name is mangled, when you list dir(anyobject), it shows classname appended to actual variable name. For ex.
  
  {% highlight Python %}
  self.__xyz = 1 # inside class A
  print A().__xyz # throws error as it is not defined.
  print A()._A__xyz # gives the __xyz value.
  {% endhighlight %}
  * Leading and training double underscore - Python reserved methods and variable, don't use this standard.
  * Single underscore - temporary variable from interpreter.
  \>>> 20 + 3
  23
  \>>> _
  23

### Variable
{% highlight Python %}
xyz ="Something"
xyz = 1  
{% endhighlight  %}
Everything in Python is object of some class and each variable is just a reference of size 8 byte (can be different in different implementation of Python). But there are some key datatypes defined by Python.
## DataTypes
The sizes of each datatype can be different based on different implementation of python.
* Numbers - 
  * int - Unlimited size
  * float - accurate upto 15 decimal
  * complex 
* Sequence - collections.abc.Sequence
  * list
  * tuple
  * range
  * string
  * bytes
  * bytearray
  * memoryview
* Set
  * set - mutable
  * frozenset - immutable
* Mapping
  * dict

### Number Operations.
* x + y, x - y, x * y, x / y, x % y, x ** y or pow(x,y)  
* x // y - Floored quotient
* abs(x)
* int(x)
* float(x)
* round(x,y) - round upto y digits
* math.floor(x), math.ceil(x)

**Bitwise Operators**
* x | y, x & y, ~x
* x ^ y - exclusive or or x and y
* x >> n, x << n shift n bits
 
### Sequence type Operations. 
* x in s, x not in s 
* s + t - concat
* s * n or n * s - repeat itself by n times.
* s[i], s[i:j], s[i:j:k] - k is steps
* len(s), min(s), max(s)
* s.index(x[, i[, j]]) - index of first x in i to j slice.
* s.count(x)

**Mutable sequence Operations- List**
All above are there, here are some more.
* s[i] = x, s[i,j] = x, s[i,j,k] = x
* s[i] = x, s[i,j] = x, s[i,j,k] = x
* del s[i:j], del s[i:j:k]
* s.append(x) - add object as it is
* s.extend(x) - iterate over x and adds
* s.clear(x) or del s[:], s.copy(x) or s[:] - shallow copy
* s *= n -Repeat n time and update itself.
* s.insert(i, x) or s[i:i]=x, s.pop([i]), s.remove(x) - remove first x
* s.reverse()

### Set Operator
* len(s), copy() - shallow copy
* x in s, x not in s
* isdisjoint(t)
* issubset(t), s <= t, s < t - proper subset
* isuperset(t),  s >= t, s > t - proper superset
* union(\*t) or s | t, intersection(\*t)  or s & t, difference  or s - t, symmetric_difference  or s ^ t, 

**Extra immutable operators**
* update(\*t) or s |=t, interset_update(\*t) or s &= t
* difference_update(\*t) or s -=t, symetric_difference_update(\*t) or s ^= t
* remove(a) - throws error if not present, discard(a)
* add(a), pop(), clear()

### Dict operator
* len(), clear(), copy(), get(k), pop(k), popitem() - removes last item
* d[key], d[key] = value, del d[key]
* key in d, key not in d
* iter(d) or iter(d.keys())
* update([other]) - overrides with other key value pair.
* items(), keys(), values() -return view and gets update when dict changes.

### Python Module
Module in python are files which can some statements, methods, variable etc. Moduels can import another modules as well. You can reload the module using {% ihighlight Python %}
  import importlib; importlib.reload(modulename)
{% endihighlight  %}.
**Execute Modules** - >>>python module1.py <arguments>
Access module parameter by, int(sys.argv[1]). Main module should have name \_\_main__.py.

**Packages** - Packages are folder with _\_init__.py files, these file can be empty but can also used to set \_\_all__ variable. Use package with dot. 

### Inheritance.
Private attribute are defined by double underscore and protected are defined by single single underscore. Private can be accesses by mangling the attribute name. Logically anything defined inside functon with self is private. For ex.
{% highlight Python %}
  class Person:
  def \_\_init__(self): 
    self.firstname = 'xyz'
    self.\_\_lastname = 'abc'
  ...

  P = Person()
  print(P.__lastname) # will throw, #AttributeError: 'Person' object has no attribute '__lastname'
{% endhighlight  %}

## Python Magic methods.
* Constructor, intializer and destructor
_\_new__(cls, [...), _\_init__(self, [...), _\_del__(self)
* Comparison magic methods
_\_cmp__(self, other), _\_eq__, _\_ne__,_\_lt__,_\_gt__,_\_le__,_\_ge__
* Unary operator
_\_pos__,_\_neg__,_\_abs__,_\_invert__,_\_round__,_\_floor__,_\_ceil__,_\_trunk__
* Arithmetic Operator
_\_add__(self, other),_\_sub__,_\_mul__,_\_div__,_\_mod__,_\_pow__,_\_lshift__,_\_and__,_\_or__,_\_xor__
* Reflected Arithmetic Operator - Switch operator and operand position (a+b to b+a). Just add 'r' to all arithmetic operator.
* Augmented assignment - (a+=b is a = a+b), Just add 'i' to all arithmetic operator.
* Conversion
_\_int__(self),_\_float__,long,complex,oct,hex,index,trunc
* Represent your class
_\_str__(self),repr,unicode,format,hash,nonzero,dir,,sizeof
* Attribute acess.
_\_getattr__(self, name), _\_setattr__(self, name, value),delattr
{% highlight JAVA %}
  def __setattr__(self, name, value):
    self.__dict__[name] = value #right
    self.name = value # wrong, will cause recursion
{% endhighlight  %}
* Custom sequence
{% highlight Python %}
  class FunctionalList:
    '''A class wrapping a list with some extra functional magic, like head,
    tail, init, last, drop, and take.'''

    def __init__(self, values=None):
        if values is None:
            self.values = []
        else:
            self.values = values

    def __len__(self):
        return len(self.values)

    def __getitem__(self, key):
        # if key is of invalid type or value, the list values will raise the error
        return self.values[key]

    def __setitem__(self, key, value):
        self.values[key] = value

    def __delitem__(self, key):
        del self.values[key]

    def __iter__(self):
        return iter(self.values)

    def __reversed__(self):
        return reversed(self.values)

    def append(self, value):
        self.values.append(value)
    def head(self):
        # get the first element
        return self.values[0]
    def tail(self):
        # get all elements after the first
        return self.values[1:]
    def init(self):
        # get elements up to the last
        return self.values[:-1]
    def last(self):
        # get last element
        return self.values[-1]
    def drop(self, n):
        # get all elements except first n
        return self.values[n:]
    def take(self, n):
        # get first n elements
        return self.values[:n]
{% endhighlight  %}
* Copying
__copy__(self),__deepcopy__(self, memodict={})
* There some more but for advance usage.

## Python specials
### Sorting classes.
* By implementing __lt__
* By sort and key 
{% highlight PYTHON%}x.sort(key=operator.attrgetter('score')){% endhighlight  %}

### Iterator
Taken from stackoverflow, For example 
{% highlight PYTHON%}
class Counter:
    def __init__(self, low, high):
        self.current = low - 1
        self.high = high

    def __iter__(self):
        return self

    def __next__(self): # Python 2: def next(self)
        self.current += 1
        if self.current < self.high:
            return self.current
        raise StopIteration


for c in Counter(3, 9):
    print(c)
{% endhighlight %}

### Generator
There is other type of iterator as well, this doesn need a full fledged class. 
{% highlight PYTHON%}
def counter(low, high):
    current = low
    while current < high:
        yield current
        current += 1

for c in counter(3, 9):
    print(c)
{% endhighlight %}

## Inheritance.
{% highlight Python %}
  class DerivedClassName(BaseClassName1,BaseClassName2..):
    pass
{% endhighlight  %}

**isinstance and issubclass** - Tells weather object belong to inhertance hierarchy and a class is subclass of something.

**super()** - Use this to point to super class. 

### Exception handling. 
We have following keywords, try, except, else, raise,finally.
{% highlight Python %}
try:
  print("Hello")
except AppError as error:
  print("Something went wrong",error)
  error_type, error_instance, traceback = sys.exc_info()
else:
  print("Nothing went wrong")
finally:
  print("The 'try except' is finished")
raise ValueError('A very specific bad thing happened.')
{% endhighlight  %}

### Lambda
{% highlight JAVA %}
  x = lambda a, b : a * b
  print(x(5, 6))
  def myfunc(n):
    return lambda a : a * n #returning lambda functions
  mydoubler = myfunc(2)
  print(mydoubler(11))
{% endhighlight  %}

### Global and nonlocal variable
nonlocal points to outer scope variable while global is for outermost which is global.
{% highlight Python %}
  x = 0
  def outer():
      x = 1
      def inner():
          nonlocal x # Output 2 if it would have been global x
          x = 2
          print("inner:", x)

      inner()
      print("outer:", x)

  outer()
  print("global:", x)

  # inner: 2
  # outer: 2
  # global: 0

  # output2
  # inner: 2
  # outer: 1
  # global: 2
{% endhighlight  %}

### Compound statements.
{% highlight Python %}
  if <expr>:
    <statement(s)>
  elif <expr>:
    <statement(s)>
  else:
    <statement(s)>
  <expr1> if <conditional_expr> else <expr2>
....
  while <expr>:
    <statement(s)>
  else:
    <additional_statement(s)>
....
for x in fruits:
  print(x)
for x in range(2, 30, 3):
  print(x)
....
class controlled_execution:
  def __enter__(self):
      set things up
      return thing
  def __exit__(self, type, value, traceback):
      tear things down

with controlled_execution() as thing:
    some code

with open("x.txt") as f:
    data = f.read()
    do something with data
{% endhighlight  %}

## Plots
Type of plots.

**Plot** - Plot with lines.
matplotlib.pyplot.plot(*args, scalex=True, scaley=True, data=None, **kwargs)
plot([x], y, [fmt], *, data=None, **kwargs)
plot([x], y, [fmt], [x2], y2, [fmt2], ..., **kwargs)

**Subplot** - Plot over plot
matplotlib.pyplot.subplot(*args, **kwargs)
subplot(nrows, ncols, index, **kwargs)
subplot(pos, **kwargs)
subplot(ax)

**Contour**
Three dimension in 2D with use of color
matplotlib.pyplot.contour(*args, data=None, **kwargs)[source]
contour([X, Y,] Z, [levels], **kwargs)

**Histogram** - Used for intervals.
matplotlib.pyplot.hist(x, bins=None, range=None, density=None, weights=None, cumulative=False, bottom=None, histtype='bar', align='mid', orientation='vertical', rwidth=None, log=False, color=None, label=None, stacked=False, normed=None, *, data=None, **kwargs)

**Bar Chart** Same as historgram but is spaced out between bars.
matplotlib.pyplot.bar(x, height, width=0.8, bottom=None, *, align='center', data=None, **kwargs)

**Pie Chart** - Something which somes to 1.
matplotlib.pyplot.pie(x, explode=None, labels=None, colors=None, autopct=None, pctdistance=0.6, shadow=False, labeldistance=1.1, startangle=None, radius=None, counterclock=True, wedgeprops=None, textprops=None, center=(0, 0), frame=False, rotatelabels=False, *, data=None)

**Scatter Plot** - Plots points.
matplotlib.pyplot.scatter(x, y, s=None, c=None, marker=None, cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None, verts=None, edgecolors=None, *, plotnonfinite=False, data=None, **kwargs
