---
layout: post
title: Spring Containers.
date: 2018-12-11 18:57:06.000000000 -05:00
type: post
parent_id: '0'
published: true
password: ''
status: publish
categories: []
tags: []
randomImage: '4'
meta:
  _edit_last: '1'
author:
  login: sanjaypatel2525
  email: sanjaypatel2525@gmail.com
  display_name: sanjaypatel2525
  first_name: ''
  last_name: ''
---
Spring IoC container is heart of the spring framework. It is container to hold the spring objects and maintaince the spring objects lifecycle. 

Spring provide two different type if containers.
* Bean Factory Container
* ApplicationContext Container

# BeanFactory Container
Here is how to create a bean and use it in java class.
{% highlight xml %}
<bean id = "..." class = "..." />
{% endhighlight %}
{% highlight java %}
InputStream is = new FileInputStream("beans.xml");
BeanFactory factory = new XmlBeanFactory(is);
//Get the bean from XML
MyClass obj = (MyClass) factory.getBean("myClassObj");
{% endhighlight %}

## BeanFactory Methods. 
* boolean containsBean(String)
* Object getBean(String)
* Object getBean(String, Class)
* Class getType(String name)
* boolean isSingleton(String)
* String[] getAliases(String)

# ApplicationContext container
This is more towards enterprise implementations, It has many extensible feature like AOP, Databases, logging, etc. It contains all the funcationality of BeanFactory as well. It's an interface "org.springframework.context.ApplicationContext";


## Common implementation are,
* FileSystemXmlApplicationContext
* ClassPathXmlApplicationContext
* WebXmlApplicationContext 

{% highlight java %}
ApplicationContext appContext = new FileSystemXmlApplicationContext("beans.xml");
MyClass obj = (MyClass) context.getBean("myClassObj");
{% endhighlight %}