---
layout: post
title: Java lambda and functional programming - Part 2
date: 2018-09-15 23:18:31.000000000 -04:00
type: post
parent_id: '0'
published: true
randomImage: '24'
password: ''
status: publish
redirect_from:
  - /2018/09/15/java-lambda-functional-programming-part-2/
  - /2018/09/15/java-lambda-functional-programming-part-2/amp/
categories:
- Java
tags:
- Java
- Java8
- Lambda
meta:
  _edit_last: '1'
  _wp_page_template: default
  _yoast_wpseo_focuskw_text_input: Java lambda and functional programming
  _yoast_wpseo_focuskw: Java lambda and functional programming
  _yoast_wpseo_metadesc: Java lambda expression Java lambda expressions are syntactical
    sugars, It is a way of reducing lines of coded. In other words, trim down the
    code and increase readability. In this blog, we will be discussing Java lambda
    and functional programming. AddInt addFunc = (a,b) -> a+b; 1 2 3 Lambda functions
    are used with f
  _yoast_wpseo_linkdex: '81'
  _yoast_wpseo_content_score: '60'
  post_grid_post_settings: a:10:{s:9:"post_skin";s:4:"flat";s:19:"custom_thumb_source";s:92:"https://abyte.stream/wp-content/plugins/post-grid/assets/frontend/css/images/placeholder.png";s:17:"font_awesome_icon";s:0:"";s:23:"font_awesome_icon_color";s:7:"#737272";s:22:"font_awesome_icon_size";s:4:"50px";s:17:"custom_youtube_id";s:0:"";s:15:"custom_vimeo_id";s:0:"";s:21:"custom_dailymotion_id";s:0:"";s:14:"custom_mp3_url";s:0:"";s:20:"custom_soundcloud_id";s:0:"";}
  _the_champ_meta: a:5:{s:7:"sharing";i:0;s:16:"vertical_sharing";i:0;s:7:"counter";i:0;s:16:"vertical_counter";i:0;s:11:"fb_comments";i:0;}
  _yoast_wpseo_primary_category: '4'
  ampforwp_custom_content_editor: ''
  ampforwp_custom_content_editor_checkbox: ''
  ampforwp-amp-on-off: default
author:
  login: sanjaypatel2525
  email: sanjaypatel2525@gmail.com
  display_name: sanjaypatel2525
  first_name: ''
  last_name: ''
---
## Java lambda and functional programming
Java lambda expressions are syntactical sugars, It is a way of reducing lines of coded.  In other words, trim down the code and increase readability. In this blog, we will be discussing Java lambda and functional programming.
{% highlight java %}AddInt addFunc = (a,b) -> a+b;
1                  2       3{% endhighlight %}
Lambda functions are used with functional interfaces in Java. There three important rules to write lambda.

The class/interface (1) should only one abstract method.
The argument for the abstract method should match to lambda before arrow part(2).
The return type(3) should match to abstract method return type.

### Lambda expression.


### Zero parameters.
Zero parameters is just an empty parenthesis. The function's name and object doesn't need to be the same, it can be different.
{% highlight java %}interface FindRandom{ 
    int random();
}

FindRandom randomFunc = () -> Maths.random();

randomFunc.random() // return a random number{% endhighlight %}


### One parameter.
One parameter do not require parenthesis. There is a Java provided a standard interface for these kinds of interface, Which accepts and returns only one argument.  Please check Functional<T,R> interface. With this standard interface, we don't need to create any new interfaces anytime for any methods which accept and return only one variable.
{% highlight java %}interface PrintSquare{ 
    int square(int a);
}

PrintSquare squareFunc = a -> a*a;

squareFunc.square(3);  // return 9{% endhighlight %}


### Multiple parameters.
Multiple parameters are enclosed in parenthesis.  Function interface having method which accepts two parameters and returns one value are often called Two-Arity Functions. Here are the Java-defined interfaces for these kinds of methods.<br />
ToIntBiFunction, ToDoubleBiFunction, ToIntBiFunction, and ToLongBiFunction
{% highlight java %}interface AddInt{ 
    int add(int a, int b);
}

AddInt add = (a,b) -> a+b;

add.add(4,5); // return 9{% endhighlight %}


### Multiple lines of code in functions body.
We add curly braces in case there are multiple lines in the body. It properly defines the starting and ending of the block.
{% highlight java %}interface SquareAndAdd{ 
    int squareAdd(int a, int b);
}

SquareAndAdd func = (a,b) -> {
    a = a*a;
    b = b*b;
    return a+b;
};

func.squareAdd(2,3); // returns 13{% endhighlight %}


### Pass Lambda function as a variable.
In Javascript we have can pass callback function to a function as below.
{% highlight java %}//javascript
var callback = function(int a){
   console.log(a);
}

var mainFunc = function(var i, var j, var func){
   var k = i +j;
   callback(k);
}

mainFunc(2,4,callback); // Here callback is passed as parameter and will be execute in form inside mainFunc{% endhighlight %}
With the functional interface and lambda expression in Java8 we can do the same.
{% highlight java %}interface DoSquare{ 
  int square(int a);
}

interface AddIntThenSquare{ 
  int add(int a, int b, DoSquare doSqaure);
}

DoSquare squareFunc = a -> a*a;
AddIntThenSquare addFunc = (a,b,c) -> {
    int k = a+b;
    return c.square(k); 
};

System.out.println(addFunc.add(2, 3, squareFunc)); //Prints 25
{% endhighlight %}


### Return Lambda function from the function
In the below example, we have a function which takes a string parameter and returns a method(functional interface).
{% highlight java %}interface Calculator{  
  OperationWrapper getOperationWrapper(String type);
}

interface OperationWrapper{
  int operate(int a, int b);
}


Calculator c = type -> {
  switch(type){
    case "add": 
      return (a,b) -> a+b;
    case "substract": 
      return (a,b) -> a-b;
    case "multiply": 
      return (a,b) -> a*b;
    default: 
      return (a,b) -> a/b;
  }
};
    
System.out.println(c.getOperationWrapper("add").operate(1, 2)); //prints 3
System.out.println(c.getOperationWrapper("substract").operate(1, 2)); //prints -1
System.out.println(c.getOperationWrapper("mulitply").operate(1, 2)); //prints 2
System.out.println(c.getOperationWrapper("devide").operate(1, 2)); //prints 0



{% endhighlight %}
There is a little more to lambda functions, Double colon operator. It looks something like this.

It might look a little bit confusing but it fairly simple and easy to understand. This is it on our Java lambda and functional programming - Part 2. Please check out next blog in this series <a href="https://abyte.stream/2018/09/15/java-8-double-colon-operator/">Java 8 double colon operator.</a>

