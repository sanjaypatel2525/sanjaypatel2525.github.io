---
layout: post
title: Spring Events and Listners.
date: 2018-12-11 18:57:06.000000000 -05:00
type: post
parent_id: '0'
published: true
password: ''
status: publish
categories: [Spring]
tags: []
randomImage: '8'
meta:
  _edit_last: '1'
author:
  login: sanjaypatel2525
  email: sanjaypatel2525@gmail.com
  display_name: sanjaypatel2525
  first_name: ''
  last_name: ''
---
Spring has inbuilt support for publishing and listing events. 
In layman term if you want to understand what is event and listener is then, 
you can think as somebody shouts your name and said hi and you listener to it and 
you want to give a reply or not it's upto you. Here you become a listener listening on your name event.

Spring provides following interfaces and equivalent annotations which you can implement to make event and listener classes.
* ApplicationEvent or //simply nothing any object can be published from Spring 4.2+
* ApplicationEventPublisher
* ApplicationListner or @EventListener

There three parties classes in Event and listener mechanism. SomeEvent, SomeEventListner, SomeEventData

### Old way
{% highlight JAVA %}
class SomeEventData{
    ...
    private String data;
    private Long dataLength;
    ...
}

class SomeEvent implements ApplicationEvent{
    private SomeEventData someEventData;
    public SomeEventData(SomeEventData someEventData){
        this.someEventData = someEventData;
    }

    public SomeEventData getSomeEventData(){
        return this.someEventData;
    }
}

class SomeEventListner implements ApplicationListner{
    @Override
    public void onApplicationEvent(SomeEvent someEvent) {
        --process(someEvent.getSomeEventData());
    }
}

class Main{
    @Autowire
    ApplicationEventPublisher applicationEventPublisher;
    public static void main(String[] args){
        ...
        applicationEventPublisher..publishEvent(new SomeEvent(new SomeEventData()));
    }
}

{% endhighlight %}

### new way
{% highlight JAVA %}
class SomeEventData{
    ...
    private String data;
    private Long dataLength;
    ...
}
// No need to implement ApplicationEvent interface
class SomeEvent {
    private SomeEventData someEventData;
    public SomeEventData(SomeEventData someEventData){
        this.someEventData = someEventData;
    }

    public SomeEventData getSomeEventData(){
        return this.someEventData;
    }
}

// No need implement   ApplicationListner
class SomeEventListner{
    @EventListener   //1
    public void handleEvent(SomeEvent someEvent) {
        --process(someEvent.getSomeEventData());
    }

    // or even publish a new event  --2
    @EventListener
    public SomeOtherEvent handleEvent(SomeEvent someEvent) {
        --process(someEvent.getSomeEventData());
        return SomeOtherEvent(new SomeOtherEventDate());
    }

    // Execute handleEvent in non blocking way  --3
    @Async 
    @EventListener
    public void handleEvent(SomeEvent someEvent) {
        --process(someEvent.getSomeEventData());
    }

    // Or use Spring expression language based condition  --4
    @EventListener(condition = "#someEvent.getDataLength()>10")
    public void handleEvent(SomeEvent someEvent) {
        --process(someEvent.getSomeEventData());
    }
}

class Main{
    @Autowire
    ApplicationEventPublisher applicationEventPublisher;
    public static void main(String[] args){
        ...
        applicationEventPublisher..publishEvent(new SomeEvent(new SomeEventData()));
    }
}

{% endhighlight %}

You can see above we have two way of defining Event and listeners. There can be n number of events and n number of listeners.
Whenever a event is published all registered event are notified and based on matching type of Event class they are executed. 

In the new example we have some points.
1. @EventListener replaces the ApplicationListner way of implementation, here within single class you can have n number of listeners.
2. @EventListener methods can return other event object and they will be automatically published. 
3. @Asyn runs the method in a new thread in a nonblocking way.
4. Condition uses SPel with this method will be called only when the inner condition satisfies. 