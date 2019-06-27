---
layout: post
title: When to use Java Optional Class
date: 2018-09-23 01:29:34.000000000 -04:00
type: post
parent_id: '0'
randomImage: '27'
published: true
password: ''
status: publish
redirect_from:
  - /2018/09/23/when-to-use-java-optional-class/
  - /2018/09/23/when-to-use-java-optional-class/amp/
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
  _yoast_wpseo_focuskw: When to use Java Optional Class
  _yoast_wpseo_linkdex: '84'
  _yoast_wpseo_content_score: '90'
  _yoast_wpseo_primary_category: '4'
  ampforwp_custom_content_editor: ''
  ampforwp_custom_content_editor_checkbox: ''
  ampforwp-amp-on-off: default
  _yoast_wpseo_metadesc: Java 8  has provided a very strong feature to handle null.
    Optional should it be used everywhere? we get a question about When to use Java
    Optional Class.
author:
  login: sanjaypatel2525
  email: sanjaypatel2525@gmail.com
  display_name: sanjaypatel2525
  first_name: ''
  last_name: ''
---
Java 8  has provided a very strong feature to write code in a better and cleaner way. This can be used to easily replace our null but should it be used everywhere? And with this, we get a question about When to use Java Optional Class.
## 1. What is Java Optional Class?
Java optional is a generic class, It is a wrapper around another classes object and provides a set of methods.
{% highlight java %}package java.util;
...
public final class Optional<T> {
    ...
    private final T value;
    ...
    public static <T> Optional<T> of(T value) {
        return new Optional<>(value);
    }
    public T get() {
        if (value == null) {
            throw new NoSuchElementException("No value present");
        }
        return value;
    }
    public boolean isPresent() {
        return value != null;
    }
   ...
}
{% endhighlight %}
As you can it has variable value and later that value can be retrieved using get.  It has some other important methods which we will discuss later in this blog. Here is a basic example of how we can use Java Optional class.
{% highlight java %}class Country{  
  private String name = "some name";
  ...
}
//Using null check -Method 1
Country country = new Country();
    
if(country != null){
    System.out.println(country.getName());
}

//Using optional Class -- Method 2
Country country = new Country();		
Optional<Country> optionalOfCountry= Optional.ofNullable(country);
if(optionalOfCountry.isPresent()){
    System.out.println(optionalOfCountry.get().getName());
}{% endhighlight %}
So what the difference between method one and method two, Method 2 doesn't require 'null' keyword but does it look like it has reduced your code and it's better to read( Different people different opinion)? I feel it's waste of time to use method 2 here. Then,
## 2. When to use Java Optional Class?
Let's start by example.
{% highlight java %}class Country{  
  State state;
        ...
}

class State{  
  District district;
        ...
}

class District{  
  Village village;
        ...
}

class Village{  
  String villageName = "default name";
        ...
}

Country country = new Country();


//Using null check -Method 1
if(country !=null){
  if(country.getState() !=null){
    if(country.getState().getDistrict() != null){
      if(country.getState().getDistrict() != null){
        if(country.getState().getDistrict().getVillage() != null){
          if(country.getState().getDistrict().getVillage().getVillageName() != null){
            System.out.println(country.getState().getDistrict().getVillage().getVillageName());
          }
          else{
            System.out.println("No village name defined");
          }
        }
      }
    }
  }
}

//Using optional Class and map function -Method 2
Optional<Country> optionalOfCountry= Optional.ofNullable(country);
System.out.println(optionalOfCountry.map(Country::getState)
    .map(State::getDistrict)
    .map(District::getVillage)
    .map(Village::getVillageName)
    .orElse("No village name defined"));{% endhighlight %}
Here we created nested classes and method 1 shows how to null check and get the end value while method 2 use Optional class and its map method to transform the input to an output variable, In the end, we used orElse method to return if everything is null.<br />
Here definitely we can say Optional way handling nulls is more readable and less code. If we conclude we can single whenever there is nested null checks better to wrap the object in Java Optional class and use it rather then writing ugly null check code and in case of one time null check use our oldy goldy null check.
Let discuss more on Optional class.
## 3. Java Optional Class Instantiation
### 3.1 Initiate an empty Optional
This is used when you want to start with an empty/null value Optional wrapper.
{% highlight java %}Optional<T> optional = Optional.empty();{% endhighlight %}
### 3.2 Initiate Optional with an object.
This is used when you want to initialize wrapper with an object, but object can't be null.
{% highlight java %}Optional<T> optional = Optional.of(T t);{% endhighlight %}
### 3.3 Initiate Optional with a nullable object.
This is used when you want to initialize wrapper with an object and object can be null.
{% highlight java %}Optional<T> optional = Optional.ofNullable(T t);{% endhighlight %}
##3. Java Optional Class Methods
### 3.3 Get the value of Optional.
This is used when you want to initialize wrapper with an object and object can be null.
{% highlight java %}T t = optional.get() ;{% endhighlight %}
### 3.3 Check if the value is present in Optional.
This is used when you want to initialize wrapper with an object and object can be null.
{% highlight java %}boolean isPresent = optional.isPresent();{% endhighlight %}
### 3.3 Call consumer if the value is present.
This is used for terminal operations,  what to do if the value is there probably print it. A consumer is a functional interface having only one method which accepts a value and returns nothing.
{% highlight java %}ifPresent(Consumer<? super T> consumer);

Optional<String> nameOptional = Optional.of("sanjay");
nameOptional.ifPresent(System.out::println);{% endhighlight %}
### 3.3 Conditional check.
Filters are used to run some predicates on Optional value which return either true or false. Based on predicate result it returns either the same Optional value or empty Optional.
{% highlight java %}Optional<T> t = optional.filter(Predicate<? super T> predicate);

Optional<Integer> optionalInt = Optional.of(24);
optionalInt = optionalInt.filter(i -> i>0); // the condition returns true so the end value remains same.

optionalInt = optionalInt.filter(i- > i<0); // it return false the end result will be equal to optional.empty();

{% endhighlight %}
### 3.3 Map Optional value to another Optional
Map function takes a <a href="https://abyte.stream/2018/09/15/java-lambda-procedural-programming/">functional interface</a> which takes T parameter and return R another parameter.
{% highlight java %}Optional<U> u = optional. map(Function<? super T, ? extends U> mapper); 

Optional<String> optString = Optional.of("sanjay");
Optional<Int> optInt = optString.map(String::length);
optInt.get(); // return 5{% endhighlight %}
### 3.3 If null then return another value
This function takes a parameter and returns the passed parameter when the optional value is null.
{% highlight java %}T t = optional. orElse(T other);{% endhighlight %}
### 3.3 If null then return the value generated by the supplier.
This is used when you want to initialize wrapper with an object and object can be null.
{% highlight java %}T t = optional. orElseGet(Supplier<? extends T> other); 

Optional<String> optString = Optional.empty();
double a = optString.orElseGet(Maths:random); //return random value{% endhighlight %}
## 4. Conclusion
In this blog, we have discussed when to use Java Optional class and what are the important method in this class. This is it about the blog When to use Java Optional Class.
