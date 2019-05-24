---
layout: post
title: Spring Autowire.
date: 2018-12-11 18:57:06.000000000 -05:00
type: post
parent_id: '0'
published: true
password: ''
status: publish
categories: []
tags: []
cardImage: Untitled.jpg
meta:
  _edit_last: '1'
author:
  login: sanjaypatel2525
  email: sanjaypatel2525@gmail.com
  display_name: sanjaypatel2525
  first_name: ''
  last_name: ''
---
Spring container has capbility of automatically injecting dependencies while intiating bean. There are 5 different modes. 
* No Autowire - XML Based Default
* Autowire by Type - Java Based Default
* Autowire by Name
* Autowire by Constructor
* Autodetect - Deprecated

You can turn on autowire feature by using following code in your XML. With this you can start using @Autowire annotation in your Java file. 
{% highlight XML %}
<context:annotation-config />
<!-- Or -->
<bean class ="org.springframework.beans.factory.annotation.AutowiredAnnotationBeanPostProcessor"/>

<bean id="application" class="com.websystique.spring.domain.Application" autowire="byName"/> 
 <!-- Application class has a applicationUser property and it will be autowired by name as there is one applicationUser Object is defined below -->

<bean id="applicationUser" class="com.websystique.spring.domain.ApplicationUser" >
    <property name="name" value="superUser"/>
</bean>

<!-- Second object ApplicationUser-->
<bean id="applicationUser2" class="com.websystique.spring.domain.ApplicationUser" >
    <property name="name" value="superUser"/>
</bean>
{% endhighlight %}

You can autowire class properties directly even if they are private, you can autowire fields by constructor paramters or setters parameter. What if you have two ApplicationUser object in container which one application object will use, In XML based you can just give objectname but in Java class you will have to use  @Qualifier("applicationUser2") to identify which one to pick.
{% highlight JAVA %}
@Autowired
@Qualifier ("applicationUser2")
private ApplicationUser applicationUser;
{% endhighlight %}

What if container doesn't find the matching object? It will throw error, you can suppress this by using @Autowired (required=false)
{% highlight JAVA %}
@Autowired(required=false)
@Qualifier ("applicationUser2")
private ApplicationUser applicationUser;
{% endhighlight %}