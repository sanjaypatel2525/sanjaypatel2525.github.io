---
layout: post
title: Spring AOP and AspectJ. 
date: 2018-12-11 18:57:06.000000000 -05:00
type: post
parent_id: '0'
published: true
password: ''
status: publish
categories: []
tags: []
randomImage: '17'
meta:
  _edit_last: '1'
author:
  login: sanjaypatel2525
  email: sanjaypatel2525@gmail.com
  display_name: sanjaypatel2525
  first_name: ''
  last_name: ''
---
Aspect-oriented programming AOP is programming paradigm with which you get flexibility of adding additional code (such as logging, security etc), without modifying your existing. This increase the modularity and the injectable code can be reused in many other places. AOP concept has few important terms. 
* Aspect - Aspect is modularization of the code which defines the concerns of class, in simpler term it contains pointcuts(rule identifies places in class) and advice (the code which will be executed on joinpoint).
* PointCuts - It defines rule which determines joinpoints(places in class) where advice will be executed. Such as "execution(* com.xyz.bo.*BO.*(..))", defines all the methods in interface ending with BO under package "com.xyz.bo".
* Joinpoints - Joinpoints are candidate where advice can injected and they are matched to pointcuts rules. Fro example method call, method before, method after, method around, constructor call, constructor before, constructor after, object initialization, Field reference, Field assignment, Handler execution, Advice execution	 etc.
* Aspect -  Aspect are the piece of code which has to are call ed at matching joinpoints and pointcuts. 
* Weaving - Weaving is process of attaching the injectable code. It can be at runtime or compile time. Spring does it at runtime and AspectJ does it at compile time.

In this blog we are going to discuss on both approached Spring AOP and ApspectJ. AspectJ has set the ground for AOP programing while Spring AOP has extended the concept and made it simpler for there use. With Spring you can either go with Spring AOP or AspectJ both. In recent release Spring AOP has adapted the AspectJ annotation with that you will not have difficulty in switching from one to another. Here are some key differences between Spring AOP and Aspect J. 
* Spring AOP is runtime weaved while AspectJ Weaved at compile time. We have three different option at compile time. **Compile time, Post Compile and Load time**.
* Spring AOP is basically straight forward and has less learning curve. AspectJ is more versatile and has more feature and join point support. 
* Spring AOP has limitation, it work only when you use the spring container. 
* AspectJ requires a utility to be run in order to modify the end compiled package or it can be configured in maven goals.
* AspectJ has some performance advantage over Spring AOP as AspectJ is weaved at compile time. 

Here is an example.
{% highlight JAVA %}
@Aspect
public class CustomLogging {

   @PointCut("execution(* com.xyz.*.*(..))")
   private void allMethodsInPackage(){}

   @Before("allMethodsInPackage()")
   public void beforeAdvice(JoinPoint joinPoint){
     System.out.println("Method called is : " + joinPoint.getSignature().getName());
      System.out.println("Going to setup student profile.");
   }  

    @AfterReturning(
    pointcut = "execution(* com.xyz.bo.EmployeeBO.*(..))",
    returning= "result")
    public void afterReturningAdvice(JoinPoint joinPoint, Object result){
      System.out.println("Method called is : " + joinPoint.getSignature().getName());
      System.out.println("Employee id : " + ((Employee)result.getEmployeeId());
    }

    @AfterThrowing(
    pointcut = "execution(* com.xyz.bo.EmployeeBO.*(..))",
    throwing= "error")
    public void afterThrowingAdvice(JoinPoint joinPoint, Throwable error){
      System.out.println("Method called is : " + joinPoint.getSignature().getName());
      System.out.println("Erros is : " + error);
    }

    ...
    @Around // It will be before and after of joinpoints.
    @After // It will be after of joinpoints. 

}
{% endhighlight %}

## Enabling AOP
### Enabling Spring AOP. 
It usage the proxy "org.springframework.aop.framework.ProxyFactoryBean" to proxy our objects and helps to inject the custom methods. 
{% highlight JAVA %}
public class ProxyFactoryBean{  
  private Object target;  
  private List interceptorNames;  
...
}  

public class BeforeAdvisor implements MethodBeforeAdvice{  
    @Override  
    public void before(Method method, Object[] args, Object target)throws Throwable {  
        System.out.println("method info:"+method.getName()+" "+method.getModifiers());  
        System.out.println("target object class name: "+target.getClass().getName());  
    }  
    ...
}  
{% endhighlight  %}

{% highlight XML%}
<bean id="someClass" class="com.xyz.SomeClass"></bean>  
<bean id="beforAdvice" class="com.xyz.BeforeAdvisor"></bean>  
  
<bean id="proxy" class="org.springframework.aop.framework.ProxyFactoryBean">  
  <property name="target" ref="someClass"></property>  
  <property name="interceptorNames">  
    <list>  
      <value>beforAdvice</value>  
    </list>  
  </property>  
</bean>
{% endhighlight %}

### Enabling Spring with AspectJ
pom.xml will have following dependencies.
{% highlight XML %}
<dependency>
  <groupId>org.springframework</groupId>
  <artifactId>spring-aop</artifactId>
  <version>${spring.version}</version>
</dependency>

<dependency>
  <groupId>org.aspectj</groupId>
  <artifactId>aspectjrt</artifactId>
  <version>1.6.11</version>
</dependency>

<dependency>
  <groupId>org.aspectj</groupId>
  <artifactId>aspectjweaver</artifactId>
  <version>1.6.11</version>
</dependency>

<!-- Or just add following-->
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-aop</artifactId>
</dependency>
{% endhighlight  %}

Enable AspectJ in XML by adding **<aop:aspectj-autoproxy />** in configuration XML and  **@EnableAspectJAutoProxy** to Enable AspectJ using Java in any configuration class.