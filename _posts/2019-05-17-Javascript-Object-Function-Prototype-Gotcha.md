---
layout: post
title: Javascript Object Oriented Function Prototype  Gotcha
date: 2018-12-11 18:57:06.000000000 -05:00
type: post
parent_id: '0'
published: true
password: ''
status: publish
categories: []
tags: []
randomImage: '2'
meta:
  _edit_last: '1'
author:
  login: sanjaypatel2525
  email: sanjaypatel2525@gmail.com
  display_name: sanjaypatel2525
  first_name: ''
  last_name: ''
---
Ok, when you come from a Object Oriented based background you will always have curiosity about Javascript. Here are few points which I want to lay down before we see some code.
* Javascript has two main concepts function and object which are derived directly from Function and Object prototype. 
* 'Function' and 'Object' are prototype defined in Javascript same as above.
* The objects are first class citizen in Javascript, there are no classes. 
* The functions themselves are objects but they are callable, they are made from 'Function' Prototype.
* The function will have both the properties __proto__ and Prototype, while object will have only __proto__. Think as __proto__ on any object gives the blueprint of function/prototype it is created from. While prototype is property of function itself and is blueprint of the same.

 
{% highlight JAVASCRIPT %}
var Cat  = function(name){
    //public variable
    this.name = name;

    //private variable or closure
    var fullname = this.name + ' W';
}

var c1 = Cat("Carol");  //probably wrong way to call but still works and 'this' is passed as window. 
console.log(window.name);  // print carol

var c2 = new Cat("Karen"); // Here new object is created with Cat prototype. So a new 'this' is passed and set the values and assigns back to c2.

Cat.prototype === c1.__proto__; // prints true, they are same thing
typeof Cat; //give "function", We created Cat as a function
typeof c1; // gives "object", we create c1 as a object out of prototype(function) Cat.
typeof c1.__proto__; //gives "object"
typeof Cat.prototype; //gives "object"
typef Cat.__proto__; //gives "function, inherits from 'Function'
Cat.__proto__ === Function.prototype; //return true;

{% endhighlight  %}


### Method creation via constructor vs prototype
* You can create method using prototype of using constructor approach. //Refer the example - 5
* Prototype way of adding properties/method is faster compared to construcor approach while constructor approach has access to private variable which prototype way does not.

{% highlight JAVASCRIPT %}
// example - 5
function Class () {}
Class.prototype.calc = function (c, d) {
    return c + d;
}
var c1 = new Class();
c1.calc(1,2); // prints 3

// Using constructor approach
function Class () {
    this.calc = function (c, d) {
    return c + d;
} 
var c1 = new Class();
c1.calc(); // prints 3, no changes at all but internally constructor approach consumes more compute, while prototype changes all places at once. Constructor use this variable which gets execute for all the instances of Class but prototype updates the blueprint itself and it is shared by all.
{% endhighlight  %}

### Null and Undefined - Don't remember no issue.
* Everything is object in Javascript except primitive type string, number, bigint, boolean, null, undefined, symbol.
* string, number, bigInt, boolean, symbol have equivalent wrapper String, Number, BigInt, Boolean, Symbol.
* Primitive type are immutable. A new memory gets allocated when there is a change in value.
* null and undefined are only one which do not have object wrap.
* null is an object and is global. It points to a location.
* undefined is whenever a variable is not defined.

{% highlight JAVASCRIPT %}
typeof null; //gives "object"
typeof undefined; //gives "undefined"
null+5; //give 5, Javascript treats null as 0 for mathematical operation.
!null // gives true
null == null // true
undefined === false // gives false, undefined is not comparable to boolean values
!undefined // gives true
undefined == false // gives false, undefined is not comparable to boolean values
undefined === undefined // true
{% endhighlight  %}

### Nan - Don't remember no issue.
Special Type of Number which comes when compiler doesn't like numeric operation or you can say kind of exception.
{% highlight JAVASCRIPT %}
undefined+12 ; // gives NaN
typeOf NaN; // Number
NaN === NaN // false.  
{% endhighlight  %}

## Object Oriented Concepts
## Making thing private.
Use closure to make thing private, they will not be visible to outside world and accessed normally using public method. Closure extends the scrope of object/variables even thought there parent reference no more exists. Here, The RetainExample call was over at the line 6, 'a' might have created and destroyed, not true since a function returned from outer function, the inner function lives forever with outer functions inner soul(variable/object - lexical scope).

{% highlight JAVASCRIPT %}
function RetainExample(){
  a = 10;
  function getA(){ return a;}
  return getA;
}

var retainedValue = RetainExample(); // line - 6
retainedValue(); // print 10.  

// Closure in action for making variable private.
var Class = function() {
  var a = 10

  this.printA(){
    console.log(a);
  }
};

var c1 = new Class();
c1.printA(); // prints 10;
c1.a ; //errors out with variable not defined.
{% endhighlight %}

### Inheritance
There are mainly two ways you can inherit property of other function.  
**Using Constructor Call.** 
{% highlight JAVASCRIPT %}
function A(x){
  this.x = x;
};

function B(x,y){
  A.call(this,x);
  this.y = y;
}
var b = new B(1,2);
b.x ; // 2
b.y ; // 1
{% endhighlight %}

**Using Object.create() Method**  
You can copy the object using create, it can copy prototype as well as in the end prototype property is an object.
{% highlight JAVASCRIPT %}
B.prototype = Object.create(A.prototype); // It's faster as it works on prototype.
{% endhighlight %}