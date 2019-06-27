---
layout: post
title: Spring Bean lifecycle.
date: 2018-12-11 18:57:06.000000000 -05:00
type: post
parent_id: '0'
published: true
password: ''
status: publish
categories: []
tags: []
randomImage: '5'
meta:
  _edit_last: '1'
author:
  login: sanjaypatel2525
  email: sanjaypatel2525@gmail.com
  display_name: sanjaypatel2525
  first_name: ''
  last_name: ''
---
As we discussed Spring has a full fledged container which manages the beans. It provide many code plugin points in bean initialization process through different ways .

* Java supported @PostConstruct and @PreDestroy annotations.
* Implement BeanPostProcessor.
* Multiple different interfaces and overriding there methods.
* init() and destroy() Or @Bean(initMethod="init",destroyMethod="destroy")
* Spring event and listeners
* Spring Boot Runners

Here is rough idea behind method that will be called.
## Bean Initialize
![smiley](/assets/2019-05-12-Spring Bean life cycle-1.JPG)

## Bean Destroy
![smiley](/assets/2019-05-12-Spring Bean life cycle-2.JPG)


## Java supported @PostConstruct and @PreDestroy
This is the very simple one, you just need to annotate any method inside you Bean class with these annotations and spring will take care of calling it. @PostConstruct is called once bean is intialized an ready for use. @PreDestroy is called once conatiner tries to remove the bean. 
{% highlight java %}
class MyClass{

@PostConstruct
void someMethodPostConstruct(){...}

@PreDestroy
void someMethodPreDestroy(){...}
}
{% endhighlight %}

## Implement BeanPostProcessor

BeanPostProcessor is spring interfaces which allows you to write your own implementations of org.springframework.beans.factory.config.BeanPostProcessor.BeanPostProcessor, there can be n number of BeanPostProcessor in your application and the org.springframework.core.Orderedorder is defined by getOrder method from Ordered interface.

{% highlight java %}

import org.springframework.beans.BeansException;
import org.springframework.beans.factory.config.BeanPostProcessor;
import org.springframework.core.Ordered;

public class CustomBeanPostProcessor implements BeanPostProcessor, Ordered {
    private static Logger logger = Logger.getLogger(CustomBeanPostProcessor.class);
    private int order;

    public CustomBeanPostProcessor() {
        logger.info("Created CustomBeanPostProcessor instance");
    }

    @Override
    public Object postProcessBeforeInitialization(Object bean, String beanName)
            throws BeansException {
        logger.info("postProcessBeforeInitialization method invoked");
        return bean;
    }

    @Override
    public Object postProcessAfterInitialization(Object bean, String beanName)
            throws BeansException {
        logger.info("postProcessAfterInitialization method invoked");
        return bean;
    }

    public void setOrder(int order) {
        this.order = order;
    }

    @Override
    public int getOrder() {
        return order;
    }
}
{% endhighlight %}

## Multiple different interfaces and overriding there methods.
There are many different interface *Aware interfaces and two InitializingBean  DisposableBean interfaces. They provide some methods which has special role spring. Each Aware interface are called after some specific events. We will not discuss this in details since there are many. It can be covered in another topic.

##  init() and destroy() Or @Bean(initMethod="init",destroyMethod="destroy")
init() and destroy()are part of Spring XML based configuration and @Bean(initMethod="init",destroyMethod="destroy")are part of Spring annotation based configuration, but they both do the same thing. Here is how to use them.
XML Based.
{% highlight XML %}
<bean id="customerService" class="com.somepackage.CustomerService" init-method="init" destroy-method="destroy">
{% endhighlight %}
{% highlight JAVA %}
Annotation Based
@Configuration
public class AppConfig {
   @Bean(initMethod="init",destroyMethod="destroy")
   public MyCustomClass getCustomClassBean() {
      return new MyCustomClass(); // This bean will have init and destroy method in it.
   }
}
{% endhighlight %}

## Spring event and  Listeners
Spring emits many application events and you register listeners to them, to do that you just need implement ApplicationListener specific type of event such as below example. Here once ContextRefreshedEvent is raised by spring this listener will be called and intern it will call contextInit method of MyBean.
{% highlight JAVA %}
public class ContextRefreshListener implements ApplicationListener<ContextRefreshedEvent> {
    @Override
    public void onApplicationEvent(ContextRefreshedEvent contextRefreshedEvent) {
        contextRefreshedEvent.getApplicationContext().getBean(MyBean.class).contextInit();

    }
}  
{% endhighlight %}

## Spring Boot Runners
Spring comes with two Runner interfaces CommandLineRunner and ApplicationRunner and they are called after application is initialized.  Here is an example.
{% highlight JAVA %}
@Configuration
class Configuration{
  ...
  @Bean
    public CommandLineRunner getRunner(ApplicationContext ctx){
        return (args) -> {
            ctx.getBean(SomeClass.class).runnerInit();
        };
    }
  ...
}
{% endhighlight %}