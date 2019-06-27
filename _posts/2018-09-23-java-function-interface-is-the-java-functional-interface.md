---
layout: post
title: Java Function Interface is the Java Functional interface
date: 2018-09-23 03:35:45.000000000 -04:00
type: post
parent_id: '0'
published: true
password: ''
status: publish
randomImage: '25'
redirect_from:
  - /2018/09/15/java-lambda-functional-programming-part-2/
  - /2018/09/15/java-lambda-functional-programming-part-2/amp/
categories:
- Java
tags:
- Java
- Java8
meta:
  _edit_last: '1'
  _wp_page_template: default
  post_grid_post_settings: a:10:{s:9:"post_skin";s:4:"flat";s:19:"custom_thumb_source";s:92:"https://abyte.stream/wp-content/plugins/post-grid/assets/frontend/css/images/placeholder.png";s:17:"font_awesome_icon";s:0:"";s:23:"font_awesome_icon_color";s:7:"#737272";s:22:"font_awesome_icon_size";s:4:"50px";s:17:"custom_youtube_id";s:0:"";s:15:"custom_vimeo_id";s:0:"";s:21:"custom_dailymotion_id";s:0:"";s:14:"custom_mp3_url";s:0:"";s:20:"custom_soundcloud_id";s:0:"";}
  _the_champ_meta: a:5:{s:7:"sharing";i:0;s:16:"vertical_sharing";i:0;s:7:"counter";i:0;s:16:"vertical_counter";i:0;s:11:"fb_comments";i:0;}
  _yoast_wpseo_focuskw: Java Function Interface is the Java Functional interface
  _yoast_wpseo_title: "%%title%% %%sitename%%"
  _yoast_wpseo_metadesc: Java functional interfaces and one specific interface Function
    interface. In the end, we will see, Java Function Interface is the Java Functional
    interface.
  _yoast_wpseo_linkdex: '84'
  _yoast_wpseo_content_score: '60'
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
In this blog, we will be discussing Java functional interfaces and one specific interface Function interface. In the end, we will see, Java Function Interface is the Java Functional interface.
## 1. What is Java functional Interface?
The interface which has only one abstract method is called Java Functional interface. Why they are so important because they are small simple and moreover everywhere in Java 8 Stream API. Java functional interfaces are annotated with @FunctionalInterface. Which enforces the programmer to keep only one method for that interface and throws the compilation error.
{% highlight java %}@FunctionalInterface
interface Country{
  String getState();
  default String getName(){
    return "CountryName";
  }
}{% endhighlight %}
The above example a simple example of a functional interface. It surely has two methods but it has only one <strong>abstract method </strong>which doesn't have any definition. For more functional interface you can read <a href="https://abyte.stream/2018/09/15/java-lambda-procedural-programming/">this blog</a>.
## 2. Java Function Interface
Java Function Interface is the Java Functional interface. It has only one method and it's signature restrict the programmer to define the method in that format only. The method accepts a parameter and returns another type of variable.
{% highlight java %}@FunctionalInterface
public interface Function<T, R> {
    R apply(T t);
    ...
}
{% endhighlight %}
As per the method signature, you can see it takes a variable of type T and returns variable of type R. Below are some examples where it can be used.
{% highlight java %}//To find a square you had to create a class and a method. 
class Util{
  public static int square(int i){
    return i*i;
  }
}

Util.square(5); // return 25.

// Using Function interface.
Function<Integer,Integer> square = i -> i*i;
square.apply(5); // return 25.

//Another example
Function<String,Integer> strLength = s -> s.length(); // equivalent to String::length
strLength.apply("Sanjay") // returns 6

{% endhighlight %}
## 3. Chaining Function Interface
Function interface provides two methods two chain it to another Function interface. Depending on the method it either executes the first Function interface before or after the provided Function interface
### 3.1 Executing another Function Interface before current.
This method chains the  Function interface and executes the parameter Function interface before the caller Function.
{% highlight java %} "1,"+s;
Function<String, String> appendTwo= s -> "2, "+s;

Function<String, String> finalString = appendOne.compose(appendTwo);
finalString.apply("3"); // print 1,2,3
{% endhighlight %}
### 3.2 Executing another Function Interface after current.
This method chains the  Function interface and executes the parameter Function interface before the caller Function.
{% highlight java %} "1,"+s;
Function<String, String> appendTwo= s -> "2, "+s;

Function<String, String> finalString = appendOne.andThen(appendTwo);
finalString.apply("3"); // print 2,1,3,{% endhighlight %}
## 4. Java Function Interface is the Java Functional interface
By now you must have figured out Function Interface is actually an interface which is annotated by @funcationInterface and has an only single method to do small function based logic.
## 5. Conclusion
In this blog, we have discussed the Java function interface and what is are functional interfaces. Also, how Function interface can be chained easily.
&nbsp;
