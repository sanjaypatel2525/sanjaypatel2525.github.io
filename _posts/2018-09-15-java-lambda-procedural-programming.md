---
layout: post
title: Java lambda and functional programming - Part 1
date: 2018-09-15 23:26:33.000000000 -04:00
type: post
parent_id: '0'
published: true
password: ''
randomImage: '28'
status: publish
categories: []
tags: []
redirect_from:
  - /2018/09/15/java-lambda-procedural-programming/
  - /2018/09/15/java-lambda-procedural-programming/amp/
meta:
  _edit_last: '1'
  _wp_page_template: default
  _yoast_wpseo_focuskw_text_input: Java 8 lambda and functional programming
  _yoast_wpseo_focuskw: Java 8 lambda and functional programming
  _yoast_wpseo_metadesc: Seeing the history of Java we can easily say, Java had troubles
    adapting functional programming. Recently Oracle has made some efforts to bring
    functional programming pattern to Java. In this blog we are going to discuss about
    Java 8 lambda and functional programming. What is functional programming? Functional
    programm
  _yoast_wpseo_linkdex: '67'
  _yoast_wpseo_content_score: '90'
  post_grid_post_settings: a:10:{s:9:"post_skin";s:4:"flat";s:19:"custom_thumb_source";s:92:"https://abyte.stream/wp-content/plugins/post-grid/assets/frontend/css/images/placeholder.png";s:17:"font_awesome_icon";s:0:"";s:23:"font_awesome_icon_color";s:7:"#737272";s:22:"font_awesome_icon_size";s:4:"50px";s:17:"custom_youtube_id";s:0:"";s:15:"custom_vimeo_id";s:0:"";s:21:"custom_dailymotion_id";s:0:"";s:14:"custom_mp3_url";s:0:"";s:20:"custom_soundcloud_id";s:0:"";}
  _the_champ_meta: a:5:{s:7:"sharing";i:0;s:16:"vertical_sharing";i:0;s:7:"counter";i:0;s:16:"vertical_counter";i:0;s:11:"fb_comments";i:0;}
  _wp_old_slug: java-lambda-and-procedural-programming
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
Seeing the history of Java we can easily say, Java had troubles adapting functional programming. Recently Oracle has made some efforts to bring functional programming pattern to Java. In this blog we are going to discuss about Java  lambda and functional programming.
## What is functional programming?
Functional programming focused mainly on three points while function creation pure functions, avoid sharing state and immutable variables.<br />
In Java world methods were part of the class and they used to use instance state (object properties) and modify them. But pure function concept says method return should solely be based on method argument and it shouldn't share state. Immutable variables are never changed in whole function call they create a copy of the value, such as primitive types. Perks of functional programming function can be passed as an argument and returned same as the objects.
For example
{% highlight java %}class Utility{
   private int i=5;

   //Pure function, It does not use instance variables
   public int sumToInt(int a, int b){
       return a+b;
   }

   //Not a pure function
   public int sumWithInstance(int a){
       return a+i;
   }
}{% endhighlight %}
## How Java has added functional programming
To support functional programming Java had to support methods without a class. As you might be aware to create a method in Java we always needed a class but functional programming requires the function to be independent of class.
{% highlight java %}class Utilty{
    public int addInt(int a, int b){
        return a+b;
    }
}

Utilty utilty = new Utilty();
utilty.addInt(1, 3);{% endhighlight %}
Here just to add two integers we had to create a class and to access methods you need to create class object and access. But as you can see this function is a pure function, it doesn't use object state and return the values based on input variables. For this Java 8 introduced function interfaces.
## Java 8 lambda and functional programming
{% highlight java %}interface AddInt{  //Line1
    int addInt(int a, int b);
}

AddInt addFunc = (a,b) -> { return a+b; } //Line 2
//or
AddInt addFunc = (a,b) -> a+b; //Line 2

System.out.println(addFunc.addInt(1,2)); // prints 3{% endhighlight %}
Let's forget about the interface, what just happened with line 2. You must have seen in Javascript something like this.
{% highlight java %}//Javascript
var addFunc = function(a,b){
    return a+b;
}
//or
var addFunc = (a,b) => {return a+b;}
//or since the functions body is one liner
var addFunc = (a,b) => a+b;

console.log(addFunc(1,2)) // prints 3{% endhighlight %}
In javascript, we save function definition in a variable and call it later same even Java 8 provides the way to achieve it. The only issue remains Java doesn't have a 'var' keyword so it usage interface/abstract class with only 1 abstract method. The whole interface/class is created only for that function.
This was a quick introduction on the functional interface and harness full power of these interfaces we have lambda way coding.  Please check<a href="https://abyte.stream/2018/09/15/java-lambda-functional-programming-part-2/"> part 2</a> of this blog.
