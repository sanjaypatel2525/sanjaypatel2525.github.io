---
layout: post
title: Understanding Java Predicate Interface
date: 2018-09-23 02:28:16.000000000 -04:00
type: post
parent_id: '0'
published: true
randomImage: '26'
password: ''
status: publish
redirect_from:
  - /2018/09/23/understanding-java-predicate-interface/
  - /2018/09/23/understanding-java-predicate-interface/amp/
categories: []
tags: []
meta:
  _edit_last: '1'
  _wp_page_template: default
  post_grid_post_settings: a:10:{s:9:"post_skin";s:4:"flat";s:19:"custom_thumb_source";s:92:"https://abyte.stream/wp-content/plugins/post-grid/assets/frontend/css/images/placeholder.png";s:17:"font_awesome_icon";s:0:"";s:23:"font_awesome_icon_color";s:7:"#737272";s:22:"font_awesome_icon_size";s:4:"50px";s:17:"custom_youtube_id";s:0:"";s:15:"custom_vimeo_id";s:0:"";s:21:"custom_dailymotion_id";s:0:"";s:14:"custom_mp3_url";s:0:"";s:20:"custom_soundcloud_id";s:0:"";}
  _the_champ_meta: a:5:{s:7:"sharing";i:0;s:16:"vertical_sharing";i:0;s:7:"counter";i:0;s:16:"vertical_counter";i:0;s:11:"fb_comments";i:0;}
  _yoast_wpseo_focuskw: Understanding Java Predicate Interface
  _yoast_wpseo_metadesc: Java 8 Streams API is very robust and feature-rich API. Predicate
    is part of Java 8 Stream API. Let's start understanding Java Predicate Interface.
  _yoast_wpseo_linkdex: '92'
  _yoast_wpseo_content_score: '60'
  _yoast_wpseo_primary_category: ''
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
Java 8 Streams API is very robust and feature-rich API. It provides multiple ways to write beautiful and fewer lines of code. In that series, there is an interface Predicate Interface. Let's start understanding Java Predicate Interface.
## 1. What is Java 8 Predicate and Understanding Java Predicate Interface.
Java 8 Predicate Interface is a functional interface, By functional interface you know it will have only one abstract method and the programmer writes the body for that one abstract method as part of <a href="https://abyte.stream/2018/09/15/java-lambda-functional-programming-part-2/">Lambda expression</a>.
Predicate is a type of functional interface which takes a parameter and return a boolean result. Let's look at the Predicate Interface.
{% highlight java %}package java.util.function;
@FunctionalInterface
public interface Predicate<T> {
    boolean test(T t);

    ...
}
{% endhighlight %}
Here as you can see it has an abstract method "test" which accepts a parameter and returns a boolean value either true or false. Let's see some example.
{% highlight java %}//Use with Stream API
List<Employee> empList = getEmployeeList() // lets say some function which return list of employee, Employee object has age and name variables. 
Predicate<Employee> p1 = p -> p.getAge() > 21;

List<Employee> newEmpList = empList.stream()
  .filter(p1)
  .collect(Collectors.<Employee>toList()); // List will iterate and check for all the employee having age greater than 21{% endhighlight %}
The above filter example takes Predicate and filters the values based on predicates result. If the result is true it keeps the employee and if the result is false it discards the employee. In the end, the collect method collects it and form another list. Predicate is used with the <a href="https://abyte.stream/2018/09/23/when-to-use-java-optional-class/">Optional</a> class as well.
## 2. Chaining Predicate interfaces.
Here are the other important methods from Predicate interface.
{% highlight java %}@FunctionalInterface
public interface Predicate<T> {
    boolean test(T t);

    default Predicate<T> and(Predicate<? super T> other) {
        Objects.requireNonNull(other);
        return (t) -> test(t) &amp;&amp; other.test(t);
    }

    default Predicate<T> or(Predicate<? super T> other) {
        Objects.requireNonNull(other);
        return (t) -> test(t) || other.test(t);
    }

    default Predicate<T> negate() {
        return (t) -> !test(t);
    }

    static <T> Predicate<T> isEqual(Object targetRef) {
        return (null == targetRef)
                ? Objects::isNull
                : object -> targetRef.equals(object);
    }
    ...
}{% endhighlight %}
Predicate interface contains three more methods for logical chaining of multiple Predicates.
### 2.1 Chain Predicate using 'And' condition.
This example shows how two predicates are joined using and condition.
{% highlight java %}//Use with Stream API
List<Employee> empList = getEmployeeList();
Predicate<Employee> p1 = p -> p.getAge() > 21;
Predicate<Employee> p2 = p -> p.getAge() < 50;

Predicate<Employee> p3 = p1.and(p2);

List<Employee> newEmpList = empList.stream()
  .filter(p3)
  .collect(Collectors.<Employee>toList()); // List will iterate and check for all the employee having age greater than 21 and less than 50{% endhighlight %}
### 2.2 Chain Predicate using 'Or' condition.
This example shows how two predicates are joined using 'or' condition.
{% highlight java %}//Use with Stream API
List<Employee> empList = getEmployeeList();
Predicate<Employee> p1 = p -> p.getAge() < 21;
Predicate<Employee> p2 = p -> p.getAge() > 50;

Predicate<Employee> p3 = p1.or(p2);

List<Employee> newEmpList = empList.stream()
  .filter(p3)
  .collect(Collectors.<Employee>toList()); // List will iterate and check for all the employee having age less than 21 or greater than 50{% endhighlight %}
### 2.3 Negate Predicate.
This example shows how to negate predicates. Turn true to false and vice versa.
{% highlight java %}//Use with Stream API
List<Employee> empList = getEmployeeList();
Predicate<Employee> p1 = p -> p.getAge() > 21;

Predicate<Employee> p2 = p1.negate(p1);

List<Employee> newEmpList = empList.stream()
  .filter(p2)
  .collect(Collectors.<Employee>toList()); // List will iterate and check for all the employee having age less than 21 or equal to 21{% endhighlight %}
### 3. Conclusion
In this blog, we discussed how to use Java 8 Predicate interface and it's important method. Later we discussed on chaining of the Predicate interface. That's it on Understanding Java Predicate Interface. You can further read about Functional Interfaces.
