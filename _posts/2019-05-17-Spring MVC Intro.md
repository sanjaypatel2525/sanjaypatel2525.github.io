---
layout: post
title: Spring MVC Intro.
date: 2018-12-11 18:57:06.000000000 -05:00
type: post
parent_id: '0'
published: true
password: ''
status: publish
categories: [Spring]
tags: []
randomImage: '10'
meta:
  _edit_last: '1'
author:
  login: sanjaypatel2525
  email: sanjaypatel2525@gmail.com
  display_name: sanjaypatel2525
  first_name: ''
  last_name: ''
---
Spring has many modules to make cut short project development lifecycle. One of the most commonalty used module is Spring MVC. 
There have many different versions of Spring MVC. We are going to discuss about the most recent ones.

## What is MVC. 
MVS stands for model, view and controller. It is a design pattern mainly for web based projects. 
![smiley](/assets/2019-05-17-Spring MVC Intro-1.JPG){: .lazyload}
* Model stands for data (java data objects/ beans)
* View is display to the user (It usage model to show dynamic data)
* Controller takes request and update data/model and redirects to view.

Spring MVC entry points is a Dispatcher servlet which can be configured in web.xml. Dispatcher Servlet is responsible for all request mapping to controller and resolving the view. Controller and it's method are mapped to specific URLs these mapping are maintained in dispatcher servlet. Spring automatically does it by scanning @controller classes and @RequestMapping annotations. Later it finds out the view name and resolved the JSP or equivalent markup language to render the data. 
![smiley](/assets/2019-05-17-Spring MVC Intro-2.JPG){: .lazyload}

## Configuring Disaptcher serverl in Web.xml 
This is not required when you are using spring boot with embeded web server. 
{% highlight XML %}
    ...
    <servlet>
        <servlet-name>spring</servlet-name>
            <servlet-class>
                org.springframework.web.servlet.DispatcherServlet
            </servlet-class>
        <load-on-startup>1</load-on-startup>
    </servlet>
 
    <servlet-mapping>
        <servlet-name>spring</servlet-name>
        <url-pattern>/</url-pattern>
    </servlet-mapping>
    ...
{% endhighlight %}

Add following to pom.xml
{% highlight XML %}
<dependency>
    <groupId>org.springframework</groupId>
    <artifactId>spring-webmvc</artifactId>
    <version>4.1.4.RELEASE</version>
</dependency>
{% endhighlight %}

## XML Based Basic Configuration 
Configure component scan package and view resolver.
{% highlight XML %}
<beans xmlns="http://www.springframework.org/schema/beans"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    xmlns:context="http://www.springframework.org/schema/context"
    xsi:schemaLocation="http://www.springframework.org/schema/beans
        http://www.springframework.org/schema/beans/spring-beans-3.0.xsd
        http://www.springframework.org/schema/context/
        http://www.springframework.org/schema/context/spring-context-3.0.xsd">
 
    <context:component-scan base-package="com.somepackage.demo" />
 
    <bean class="org.springframework.web.servlet.mvc.annotation.DefaultAnnotationHandlerMapping" />
    <bean class="org.springframework.web.servlet.mvc.annotation.AnnotationMethodHandlerAdapter" />
     
    <bean class="org.springframework.web.servlet.view.InternalResourceViewResolver">
        <property name="prefix" value="/WEB-INF/views/" />
        <property name="suffix" value=".jsp" />
    </bean>
 
</beans>
{% endhighlight %}

## Spring Boot Web MVC configuration.
Add following in pom.xml
{% highlight XML %}
        <dependency>
            <!-- Import dependency management from Spring Boot -->
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-dependencies</artifactId>
            <version>2.1.3.RELEASE</version>
            <type>pom</type>
            <scope>import</scope>
        </dependency>
         <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-web</artifactId>
            <version>2.1.3.RELEASE</version>
        </dependency>
{% endhighlight %}

## Spring Boot view resolver.
{% highlight JAVA %}
@Configuration
@EnableWebMvc
public class MvcConfiguration extends WebMvcConfigurerAdapter{
    @Bean
    public ViewResolver getViewResolver() {
        InternalResourceViewResolver resolver = new InternalResourceViewResolver();
        resolver.setPrefix("/WEB-INF/");
        resolver.setSuffix(".jsp");
        return resolver;
    }  
}
{% endhighlight %}

## Create Controller and create get resource.

{% highlight JAVA %}
@Controller
@RequestMapping("/myresource")
public class MyResource {

    @GetMapping("/sayHi")
    public String sayHi(@RequestParam(name="name", required=false, defaultValue="World") String name, Model model) {
        model.addAttribute("name", name);
        return "viewPage"; // will resolve to WEB-INF/viewPage.jsp
    }

}
{% endhighlight %}

## Run Spring Boot.
{% highlight JAVA %}
@SpringBootApplication
public class Application {

    public static void main(String[] args) {
        SpringApplication.run(Application.class, args);
    }

}
{% endhighlight %}


## Spring MVC annotations.
Here we will see the Spring annotations briefly. 
* @Controller Or @RestController
* @RequestMapping
* @PathVariable
* @RequestParam
* @ModelAttribute
* @RequestBody and @ResponseBody
* @RequestHeader

### @Controller Or @RestController
Controller provide a URL to method mapping, Controller classes will be singleton and will automatically gets registered in Spring Dispatcher. RestController is specials type of controller to serialize and transform(JSON,XML etc) the response object directly. If not you would have to use **@ResponseBody** to transform the objects.

**RequestMapping**, Request mapping is used define the URL path for that method or controller class.

**@RequestBody**, In case you want deserialize the request object to some class you can use @RequestBody. It transform JSON,XML to java object. 

**@RequestHeader**, In case you want capture request headers in some variable. 
{% highlight JAVA %}
@RestController
@RequestMapping("someresource")
public class ResourceRestController {
     
    @GetMapping("/{id}", produces = "application/json")
    public Book getResource(@PathVariable int id) { // PathVariable bind the URL path value to the method parameters
        return resourceDao.getResource(id);
    }

    @GetMapping("/", produces = "application/json")
    public Book getResource(@RequestParam int id) { // RequestParam bind the request parameter value to the method parameters, such as someurl?id=value1.
        return resourceDao.getResource(id);
    }

    @PostMapping("/save", produces = "application/json", consumes = "application/json")
    public Book getResource(Book book) {
        return resourceDao.update(id);
    }
}
....
@Controller
@RequestMapping("someresource")
public class ResourceRestController {
     
    @GetMapping("/{id}", produces = "application/json")
    public @ResponseBody Book getResource(@PathVariable int id) {
        return resourceDao.getResource(id);
    }

    @PostMapping("/save", produces = "application/json")
    public @ResponseBody Book getResource(@RequestBody Book book) {
        return resourceDao.update(id);
    }

     @GetMapping("/{id}", produces = "application/json")
    public @ResponseBody Book getResource(@@RequestHeader(value="User-Agent") String userAgent) { //Request header 
        if(userAgent.equals("Mozilla")) return resourceDao.getResource(id);
        return null;
    }
}
{% endhighlight %}

## Model, ModelMap, ModelAndView, ModelAttribute
Model is an interface and ModelMap is implementation of that class you can use them interchangeably.
{% highlight JAVA %}
...
 @GetMapping("/modelExample", produces = "application/json")
    public String getResource(Model model) { //Request header 
        model.addAtttribute("msg","Hello there");
        return "viewName";
 }
    @GetMapping("/modelExampleWithView", produces = "application/json")
    public ModelAndView getResource(Model model) { //Request header 
        ModelAndView modelAndView = new ModelAndView("viewPage");
        modelAndView.addObject("msg","Hello there");
        return modelAndView;
 }

{% endhighlight %}

**ModelAttribute**, You can write a method which return some value and that value you want to store in model object.
{% highlight JAVA %}
...
@ModelAttribute  // model.get("employee") will give the value returned from this method.
public Employee getEmployee(String number) {
    return employeeDao.getEmployee(number);
}
{% endhighlight %}