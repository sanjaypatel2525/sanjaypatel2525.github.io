---
layout: post
title: Spring IoC and DI.
date: 2018-12-11 18:57:06.000000000 -05:00
type: post
parent_id: '0'
published: true
password: ''
status: publish
categories: [Spring]
tags: []
randomImage: '3'
meta:
  _edit_last: '1'
author:
  login: sanjaypatel2525
  email: sanjaypatel2525@gmail.com
  display_name: sanjaypatel2525
  first_name: ''
  last_name: ''
---
Normally objects are initialized and bound in code itself. Inversion of control is a approach to bind object at runtime using a binder program. 

IoC is design patter to give control to components, while it usage Dependency Injection DI to achieve that. DI is helps us giving framework to bind the object at runtime. Example of IoC are,
* DI
* Factory pattern
* Service locator framework

# Bean creation in Spring XML
* Constuctor 
{% highlight xml %}
<bean id="someBean"/>
{% endhighlight %}
* Static Factory method 
{% highlight xml %}
<bean id="someBean" factory-method="getSingeltonObject"/>
{% endhighlight  %}
* Another factor class and method  
{% highlight xml %}
<bean id="someBean"  factory-bean="SomoeFactoryClass" factory-method="getSingleToOfSomeFactoryClass"></bean>
{% endhighlight %}

# Bean scope
* Singleton - Will have only one thorough application.
* Prototype - Can have multiple objects. 
* Request - Will store variable on a request and recreate on new request.
* session - Will create and keep been on session. 
* global-session - consider it as application scope. Rarely changes.

{% highlight xml %}
 <bean id="someBean" class="com.packahe.DemoBean" scope="session" />
{% endhighlight %}

{% highlight java %}
@Service
@Scope("session")
public class SomeBean
{
   //Some code
}
{% endhighlight %}

