---
layout: post
title: Web Services in Java world. 
date: 2018-12-11 18:57:06.000000000 -05:00
type: post
parent_id: '0'
published: true
password: ''
status: publish
categories: []
tags: []
randomImage: '22'
meta:
  _edit_last: '1'
author:
  login: sanjaypatel2525
  email: sanjaypatel2525@gmail.com
  display_name: sanjaypatel2525
  first_name: ''
  last_name: ''
---
The journey of communication in Java has many stops. RPC, RMI, CORBA, WebServices (SOAP, RESTful), EJB Standards, ORM and JPA, JDBC etc. Let's understand all in one liner statement.
* **RPC** - Calling a remote method of a server directly is done via RPC standards. No language barrier. 
* **RMI** - RMI is mapping remote object to proxied local object and call method via proxied object. It is part of Java and deals with object. 
* **Webservice** - Webservice is comparable to RPC but it has more sophisticated structure of communicating message. There are two type of Webservice SOAP(Defines and works on WSDL) and Restful(works over entity).
* **EJB** - A common standard, set of services provided by Application servers such connection pooling, caching etc. You don't need to write code for some standard requirement some enterprise level server have inbuilt support for these services, just use them. EJB Work on top of RMI and available only for Java. 
* **ORM** - Technique to map data between different system such as database to java objects. Hibernate is an implementation of ORM.
* **JPA** - A standard and set of specification defined to communicate with different relational database. Hides low level details and provides programmer a standard way to talk to relational database.
* **JDBC** - A set of java jars provided by different database vendors to support java to database communication.  
There are many others but it gives a idea where are we going. First three are for application to application communication while last 4 mainly deals with a application and database, 4th from last EJB has bit bigger scope then database access. Let's go back to web services topic.

Webservice can be defines as way of communication between two different application (can be in different language) over the network or same system using a specific type document (Can be XML, JSON etc). There are mainly two type of webservice **SOAP**(Simple Object Access Protocol)  and **RESTful**. SOAP again has two flavours **RPC style** and **document based**. In java world SOAP is **JAX-WS** and RESTful is **JAX-RS**.

### Here are some important points. 
* SOAP is protocol (Set of rules) and REST is a design/architecture. 
* REST (Representational state transfer) is a design and has set of principles.
   * Should be stateless
   * Interface based - Should have resources and method on resources.
   * Independent Client server - Client server can change with no dependency on each other.
   * Cacheable - Resource must declare them cacheable.
   * Layered system - Server can have many layers inside it, such authentication, business logic, storage. Client need not to anything about it.
* SOAP usage service to expose business logic, Rest usage URIs. 
* SOAP is strict and defines the standards in WSDL.
* SOAP WSDL has place for extra security. REST you will have to define yourself.
* SOAP deals only with XML, REST can work with other datatype as well. 
* SOAP and REST both can work over different communication protocol such as SMTP, FTP, HTTP, custom. 
* REST has good support on HTTP but can work on other protocol as well. 

More on [SOAP](/SOAP-in-Java-world/ "SOAP") and [REST](/REST-in-Java-world/ "REST"). Please follow the links. 