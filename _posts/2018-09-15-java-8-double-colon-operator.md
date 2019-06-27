---
layout: post
title: Java 8 double colon operator
date: 2018-09-15 22:43:51.000000000 -04:00
type: post
parent_id: '0'
randomImage: '29'
published: true
password: ''
status: publish
redirect_from:
  - /2018/09/15/java-8-double-colon-operator/
  - /2018/09/15/java-8-double-colon-operator/amp/
categories:
- Java
tags:
- Java
- Java8
- Lambda
meta:
  _edit_last: '1'
  _wp_page_template: default
  post_grid_post_settings: a:10:{s:9:"post_skin";s:4:"flat";s:19:"custom_thumb_source";s:92:"https://abyte.stream/wp-content/plugins/post-grid/assets/frontend/css/images/placeholder.png";s:17:"font_awesome_icon";s:0:"";s:23:"font_awesome_icon_color";s:7:"#737272";s:22:"font_awesome_icon_size";s:4:"50px";s:17:"custom_youtube_id";s:0:"";s:15:"custom_vimeo_id";s:0:"";s:21:"custom_dailymotion_id";s:0:"";s:14:"custom_mp3_url";s:0:"";s:20:"custom_soundcloud_id";s:0:"";}
  _the_champ_meta: a:5:{s:7:"sharing";i:0;s:16:"vertical_sharing";i:0;s:7:"counter";i:0;s:16:"vertical_counter";i:0;s:11:"fb_comments";i:0;}
  _yoast_wpseo_focuskw_text_input: Java 8 double colon operator
  _yoast_wpseo_focuskw: Java 8 double colon operator
  _yoast_wpseo_linkdex: '85'
  _yoast_wpseo_content_score: '30'
  _yoast_wpseo_primary_category: '4'
  ampforwp_custom_content_editor: ''
  ampforwp_custom_content_editor_checkbox: ''
  ampforwp-amp-on-off: default
  _yoast_wpseo_metadesc: Lambda for Method reference Or Java 8 Double colon operator
    What are the Java Double colon operator? You must have seen some shortcode like,
    System.out::println Java 8 method (functional interface) are just like objects,
    They can be passed as parameters and can be returned from another method. So why
    not assign Java me
author:
  login: sanjaypatel2525
  email: sanjaypatel2525@gmail.com
  display_name: sanjaypatel2525
  first_name: ''
  last_name: ''
---
## Lambda for Method reference Or Java Double colon operator
What is the Java 8 double colon operator ? You must have seen some short code like,
{% highlight java %}System.out::println

{% endhighlight %}
In order to understand better, you must know about functional interface. Please check blog <a href="https://abyte.stream/2018/09/10/java-lambda-and-procedural-programming">Java lambda and functional programming.</a>
Java 8  method (functional interface) are just like objects, They can be passed as parameters and can be returned from another method. So why not assign Java methods to a new another variable.  Something like javascript.
{% highlight java %}var squareFunc1 = function(var x){
return x*x;
}
var squareFunc2 = squareFunc1 ;{% endhighlight %}
Here is how we can do the same in Java.
{% highlight java %}interface Utility{
 int biggerNumber(int x,int y);
}
Utility utility = Math::max; //Assign method definition to another signature matching method.

utility.biggerNumber(3,8) // returns 8
{% endhighlight %}
So above you can see, We have created an interface with only one abstract method having signature same as the signature of Math.max method. Later we have assigned the definition of Math.max to Utility. Which mean the definition is assigned is to Utility .biggerNumber. Now we can call the method as shown in the last line.
There are different ways to where we can use double operator lambda operator


### Static method reference
Assign static method to a matching functional interface. as shown in line 9
{% highlight java %}public interface StringSearch{
    public int search(String s1, String s2);
}
public class MyClass{
    public static int find(String s1, String s2){
        return s1.lastIndexOf(s2);
    }
}
StringSearch search = MyClass::find; // Line 9.{% endhighlight %}


### Parameter method reference
Here functional interface with two parameters gets the definition from a method having only one parameter. Stange !!.
{% highlight java %}public interface StringSearch{
    public int search(String s1, String s2);
}

str1.indexOf(String str2) // line 4

StringSearch search= String::indexOf;{% endhighlight %}
Not so strange, The object type on which method is called and method parameters are matched to the functional interface. In this StringSearch.search take two param and params are matched to str1 and str2 from line 4.
Instance method reference<br />
This one is the same as the static method reference.
{% highlight java %}public interface StringSearch{
    public int search(String s1, String s2);
}
public class MyClass{
    public int find(String s1, String s2){
        return s1.lastIndexOf(s2);
    }
}
MyClass myClass = new MyClass();
StringSearch search = myClass::find; // Line 9.{% endhighlight %}


### Constructor reference.
This is the one, It can be very handy in factory patterns.
{% highlight java %}public interface StringFactory {
    public String create(char[] val);
}
StringFactory factory = String::new;{% endhighlight %}


These were the few examples of how we can use Java lambda to reduce code length and make it more readable. Since these expressions represent function declaration hence these Java 8 double colon operator can be used in place of Java 8 arrow function. For example.
{% highlight java %}Utility utility = Math::max
Or 
Utility utility = (x,y) -&gt; return x &gt; y ? x: y;
{% endhighlight %}
&nbsp;
