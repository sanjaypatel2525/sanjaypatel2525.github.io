---
layout: post
title: Spring Security Architecture.
date: 2018-12-11 18:57:06.000000000 -05:00
type: post
parent_id: '0'
published: true
password: ''
status: publish
categories: [Spring]
tags: []
randomImage: '12'
meta:
  _edit_last: '1'
author:
  login: sanjaypatel2525
  email: sanjaypatel2525@gmail.com
  display_name: sanjaypatel2525
  first_name: ''
  last_name: ''
---
In case you are looking for basic spring security understating refer to the previous blog of this series [Spring Security basic](/Spring-MVC-Intro/ "Spring Security basic").

The blog we will discussing briefly on spring security architecure, in the end you will know where all you can plugin your code and improve your project security. Spring security basically provides two feature authentication based on credentials and authorization/access control based on roles. First of all let's look into below diagrams and then we will have detailed discussion on this. 
![gras](/assets/2019-05-31-Spring Security Architecture-1.png){: .lazyload}  
Here are few pointer from above images. 
* AuthenticationManager is an interface with only one method authenticate.
* ProviderManager is most used implementation of AuthenticationManager.
* ProviderManager can have multiple AuthenticationProvider, You can configure different AuthenticationProvider for different authentication mechanism. Each of them will be called in chain.

![gras](/assets/2019-05-31-Spring Security Architecture-2.png){: .lazyload}
Source: https://spring.io/guides/topicals/spring-security-architecture  
Here are few pointer from above images
* There can be n number of ProviderManager in one application, for different URL path you can assign different provider managers such 'apiv1/*, apiv2/*'.


## AuthenticationManagerBuilder
AuthenticationManagerBuilder is utility helps us configuring the AuthenticationManager, such as picking the right strategy (inmemory, jdbc, datasource etc).

## Authorization/Access control
There are two important class AccessDecisionManager and AccessDecisionVoter same as ProviderManager and AuthenticationProvider pattern. AccessDecisionManager has three implementation(AffirmativeBased, ConsensusBased,UnanimousBased) and each contains list of AccessDecisionVoter.
![gras](/assets/2019-05-31-Spring Security Architecture-3.png){: .lazyload}  
For example, AffirmativeBased implementation run all the AccessDecisionVoter and if any of them throws exception it denies the access otherwise access is granted. 

### Access User object.
In case you need to access the User the logged in user object.
{% highlight JAVA %} 
@RequestMapping("/foo")
public String foo(@AuthenticationPrincipal User user) {
  ... // do stuff with user
}
@RequestMapping("/foo")
public String foo(Principal principal) {
  Authentication authentication = (Authentication) principal;
  User = (User) authentication.getPrincipal();
  ... // do stuff with user
}
{% endhighlight %}