---
layout: post
title: Spring Java Based Configuration.
date: 2018-12-11 18:57:06.000000000 -05:00
type: post
parent_id: '0'
published: true
password: ''
status: publish
categories: [Spring]
tags: []
randomImage: '7'
meta:
  _edit_last: '1'
author:
  login: sanjaypatel2525
  email: sanjaypatel2525@gmail.com
  display_name: sanjaypatel2525
  first_name: ''
  last_name: ''
---
Spring provides two types of configuration, XML based which came in market first and then Java based. Since we already know what are the basics behind Spring configuration, anything and everything is bean. So how does spring supports that.

Let's see some code first. 
{% highlight JAVA %}
@Configuration //1
public class AppConfig {
    @Bean  // 2
    public SomeService SomeService() {
        return new SomeServiceImpl();
    }

    @Bean(name = "bar")  // 3
    public SomeService SomeService() {
        return new SomeServiceImpl();
    }

    @Bean(initMethodName = "init") //4
    public Foo foo() {
        return new Foo();
    }
    @Bean(destroyMethodName="cleanup") //5
    public Bar bar() {
        return new Bar();
    }

    @Bean
    @Scope("prototype")  //6
    public Encryptor encryptor() {
        // ...
    }

}    
{% endhighlight %}

## @Configuration annotation
This can be related to a XML file, not exactly though but it defines a class annotation with @Configuration will mainly be used to configured spring bean. Even a configuration class will have a singleton instance in spring container.

## @Bean
@Bean mostly used over getter methods which returns a object of a class. This you can relate to <bean/> tag of XML based configuration.  Bean attribute with name property will override the object name and later you can '@Autowire' it with '@qualifier(name=). initMethodName and destroyMethodName provides the feature what we use to have in Spring XML based configuration init-method and destroy-method.

## @Scope
Scope attribute helps us to define the scope of the bean which have learnt in previous blog singleton, prototype, request, session, global and global session.

## Pros of Java Based Configuration. 
* Java is type safe, you will get error if try to do something wrong.
* Refractoring code very easy.

## Pros of XML Based Configuration. 
* Centralized configuration, you don't need to search your code for a bean you can find everything in XML.
* You can Hot patch XML file easily, as these wouldn't get's converted to class file.
* You can define explicit name of bean and wire them.