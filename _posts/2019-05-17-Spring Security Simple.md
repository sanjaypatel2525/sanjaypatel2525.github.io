---
layout: post
title: Spring Security Simple.
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
This blog we are going to discuss a easiest approach to implement spring security, Spring Security Details is will more into customization points wrt to standard spring architecture. 

Spring Security focuses on two main points.
* Authentication : Verify/Identify the requester's/user's idenity.
* Authorization: Allow/block resource access based on requester/user's access level.

There are many authentication/authorization models supported in Spring such, LDAP, oauth, oauth2, OpenId, HTTP Basic etc. Also, Spring security has many feature Single such as sign-on, LDAP. Remember me etc.

### Spring XML based Configuration. 
## POM needs following depedencies. 
{% highlight XML %}
<dependency>
    <groupId>org.springframework</groupId>
    <artifactId>spring-webmvc</artifactId>
</dependency>
<dependency>
    <groupId>org.springframework.security</groupId>
    <artifactId>spring-security-web</artifactId>
</dependency>
<dependency>
    <groupId>org.springframework.security</groupId>
    <artifactId>spring-security-core</artifactId>
</dependency>
<dependency>
    <groupId>org.springframework.security</groupId>
    <artifactId>spring-security-config</artifactId>
</dependency>  
{% endhighlight %}

## web.xml
{% highlight XML %}
<filter>
    <filter-name>springSecurityFilterChain</filter-name>
    <filter-class>org.springframework.web.filter.DelegatingFilterProxy</filter-class>
</filter>
<filter-mapping>
    <filter-name>springSecurityFilterChain</filter-name>
    <url-pattern>/*</url-pattern>
</filter-mapping>
<context-param>
    <param-name>contextConfigLocation</param-name>
    <param-value>  
            /WEB-INF/spring-servlet.xml  
            /WEB-INF/spring-security.xml  
        </param-value>
</context-param>  
{% endhighlight %}

## spring-security.xml
Basic HTTP authentication. 
{% highlight XML %}
<beans:beans xmlns="http://www.springframework.org/schema/security"  
    xmlns:beans="http://www.springframework.org/schema/beans"  
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"  
    xsi:schemaLocation="http://www.springframework.org/schema/beans  
    http://www.springframework.org/schema/beans/spring-beans.xsd  
    http://www.springframework.org/schema/security  
    http://www.springframework.org/schema/security/spring-security.xsd">  
    <http auto-config="true">  
        <intercept-url pattern="/admin" access="hasRole('ROLE_ADMIN')" />  
    </http>  
    <authentication-manager>  
      <authentication-provider>  
        <user-service>  
        <user name="admin" password="password" authorities="hasRole(ROLE_ADMIN)" />  
        </user-service>  
      </authentication-provider>  
    </authentication-manager>  
</beans:beans>  
{% endhighlight %}

## Java based configuration.
### pom.xml
{% highlight XML %}
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-security</artifactId>
</dependency>
{% endhighlight %}
If add following dependency, authetnication is by default enabled, in case you want disable auto configuration use @SpringBootApplication(exclude = { SecurityAutoConfiguration.class }) at main class.

### Create configuration class.
{% highlight JAVA %}
@Configuration
@EnableWebSecurity
public class BasicConfiguration extends WebSecurityConfigurerAdapter {
 
    @Override
    protected void configure(AuthenticationManagerBuilder auth)
      throws Exception {
        auth
          .inMemoryAuthentication()
          .withUser("user1")
            .password("password1")
            .roles("USER")
            .and()
          .withUser("admin")
            .password("password")
            .roles("USER", "ADMIN");
    }
 
    @Override
    protected void configure(HttpSecurity http) throws Exception {
        http
          .authorizeRequests()
          .anyRequest()
          .authenticated()
          .and()
          .httpBasic();
    }
}
{% endhighlight %}
