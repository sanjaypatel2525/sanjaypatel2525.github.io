---
layout: post
title: Java Consumer and Supplier Interfaces
date: 2018-09-23 06:32:49.000000000 -04:00
type: post
parent_id: '0'
published: true
password: ''
randomImage: '23'
status: publish
redirect_from:
  - /2018/09/23/java-consumer-and-supplier-interfaces/
  - /2018/09/23/java-consumer-and-supplier-interfaces/amp/
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
  _yoast_wpseo_focuskw: Java Consumer and Supplier Interfaces
  _yoast_wpseo_metadesc: Java 8 has provided many single method interfaces. Java Consumer
    and Supplier Interfaces are the one of them. The log discusses about these two
    interfaces.
  _yoast_wpseo_linkdex: '81'
  _yoast_wpseo_content_score: '30'
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
In continuation of our discussion on <a href="https://abyte.stream/2018/09/15/java-lambda-functional-programming-part-2/">functional interfaces</a>, Java 8 has provided many single method interfaces. Based on the abstract method's signature Java has given them different names such as Consumer, Supplier, Function, BiFuction, Predicate etc. In this blog, we are going to discuss on Java Consumer and Supplier Interfaces.
## 1. Consumer functional interface.
Consumer functional interface is an interface having only one abstract method that accepts a single parameter and returns nothing. Here is how the interface looks like.
{% highlight java %}@FunctionalInterface
public interface Consumer<T> {
    void accept(T t);
    ...
}{% endhighlight %}
Here is an example of how Consumer interface can be used.
{% highlight java %}Consumer<String> c = x -> System.out.println(x.toLowerCase());  
c.accept("TestinG"); //Prints testing{% endhighlight %}
### 1.1 Consumer functional interface chaining
One value can be passed to multiple consumers one after one. Here is an example.
{% highlight java %}Consumer<String> c1 = (x) -> System.out.println(x.toUpperCase());
Consumer<String> c2 = (x) -> System.out.println(x.length);

c1.andThen(c2).accept("testing"); // prints "TESTING" and then "7"{% endhighlight %}
### 1.2 Primitive consumer types
Here are some primitive supplier types.
<ul>
<li>IntConsumer - Function takes one argument int and returns nothing.</li>
<li>LongConsumer - Function takes one argument Long and returns nothing.</li>
<li>DoubleConsumer - Function takes one argument Double and returns nothing.</li>
</ul>
## 3. Supplier functional interface.
Supplier functional interface is an interface having only one abstract method doesn't take any parameter but returns one value. This interface is used mainly to generate indefinite values in Java and also can be used in place of the factory patterns.
{% highlight java %}@FunctionalInterface
public interface Supplier<T> {
    T get();
}
{% endhighlight %}
&nbsp;
Here is an example of how Consumer interface can be used.
{% highlight java %}Supplier<Double> random = Math::random;
random.get(); //Prints a random number{% endhighlight %}
### 3.1 Primitive supplier types
Here are some primitive supplier types.
<ul>
<li>IntSupplier - Function doesn't take any argument and returns an int.</li>
<li>LongSupplier - Function doesn't take any argument and returns a Long.</li>
<li>DoubleSupplier - Function doesn't take any argument and returns a Double.</li>
<li>BooleanSupplier - Function doesn't take any argument and returns a Boolean</li>
</ul>
## 4. Conclusion
In this blog, we discussed two common type of interface supplier and consumer. Both the function interface are very frequently used in Java 8 Streams API.  We visited a little on primitive types of these interfaces. This is it about Java Consumer and Supplier Interfaces.
