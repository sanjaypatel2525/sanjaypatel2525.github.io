---
layout: post
title: SOAP Services in Java world. 
date: 2018-12-11 18:57:06.000000000 -05:00
type: post
parent_id: '0'
published: true
password: ''
status: publish
categories: []
tags: []
randomImage: '19'
meta:
  _edit_last: '1'
author:
  login: sanjaypatel2525
  email: sanjaypatel2525@gmail.com
  display_name: sanjaypatel2525
  first_name: ''
  last_name: ''
---
SOAP(Simple Object Access Protocol) is one of the popular way of inter application communication, application can be on different system or same system using network, can on different languages. SOAP can work with many protocol such http, SMTP etc. SOAP has two flavour RPC Style and Document based. SOAP has strict contract defined in form of WSDL. You can start creating Java class and later define WSDL based on class(bottom up approach) or define WSDL and generate Java class(top down approach). Defining WSDL can be tedious sometime for complex services so top down can be tedious compared bottom up approach.  

### What is WSDL. 
WSDL is contract definition written in form of single XML. It contains message type of request and response, endpoints etc. Here is the detailed list of WSDL component.
* **Definition** - It's root element contains all the required tags to define web service.
* **Types** - To define complex types which is formed out of basic types.
* **PortType** - It holds set of operations.
* **Operations** - Define a service abstract input and output.
* **Binding** - Binding prototype to specific protocol RPC or Document based and also defines transport http or smpt or anything else.
* **Service** - Defines the service endpoint URL and provide a little bit of writeup about service. Only for information purpose.
 
{% highlight XML %}
<definitions name = "EmployeeService"
   targetNamespace = "http://www.examples.com/wsdl/EmployeeService.wsdl"
   xmlns = "http://schemas.xmlsoap.org/wsdl/"
   xmlns:soap = "http://schemas.xmlsoap.org/wsdl/soap/"
   xmlns:tns = "http://www.examples.com/wsdl/EmployeeService.wsdl"
   xmlns:xsd = "http://www.w3.org/2001/XMLSchema">
 
   <types>
      to define complex types if there are any........
   </types>
   <message name = "GetEmployeeRequest">
      <part name = "id" type = "xsd:string"/>
   </message>
	
   <message name = "GetEmployeeResponse">
      <part name = "name" type = "xsd:string"/>
   </message>

   <portType name = "Employee_PortType">
      <operation name = "getEmployee">
         <input message = "tns:GetEmployeeRequest"/>
         <output message = "tns:GetEmployeeResponse"/>
      </operation>
   </portType>

   <binding name = "Employee_Binding" type = "tns:Employee_PortType">
      <soap:binding style = "rpc"
         transport = "http://schemas.xmlsoap.org/soap/http"/>
      <operation name = "getEmployee">
         <soap:operation soapAction = "getEmployee"/>
         <input>
            <soap:body
               encodingStyle = "http://schemas.xmlsoap.org/soap/encoding/"
               namespace = "urn:examples:employeeservice"
               use = "encoded"/>
         </input>
         <output>
            <soap:body
               encodingStyle = "http://schemas.xmlsoap.org/soap/encoding/"
               namespace = "urn:examples:employeeservice"
               use = "encoded"/>
         </output>
      </operation>
   </binding>

   <service name = "EmployeeService">
      <documentation>WSDL File for EmployeeService</documentation>
      <port binding = "tns:Employee_Binding" name = "Employee_Port">
         <soap:address
            location = "http://www.examples.com/GetEmployee/" />
      </port>
   </service>
</definitions>
{% endhighlight  %}

### RPC vs Document Based. 
These two are type of message formate. Document based is full fledged complex message which can have multiple for itself while RPC is straight forward input output parameter for a method. You can say Document follows Object based approach and easy to map to and from a Java object while, RPC based is just a simple input output parameter based on that specific method.

## Top down approach.
### Java command line tool
Java provides a small tool to generate Java classes based on defined WSDL. It can create a server side classes as well as client side also.  *wsimport -s . -p com.xyz.jaxws.server.topdown employeeservice.wsdl*  
  It will creates three classes EmployeeService.java, EmployeeService_Service.java and ObjectFactory.java
{% highlight Java %}
@WebService(
  name = "EmployeeService", 
  targetNamespace = "******")
@SOAPBinding(style=Style.RPC, use=Use.ENCODED)
@XmlSeeAlso({
    ObjectFactory.class
})
public interface EmployeeService {
    @WebMethod(
      action = "*******"
      + "EmployeeService/getEmployee")
    @WebResult(
      name = "GetEmployeeResponse", 
      targetNamespace = "********", 
      partName = "name")
    public String getEmployee(int id);
}

@WebService(endpointInterface="com.xyz.EmployeeService")
public interface EmployeeServiceImpl implements  EmployeeService{
  @Override
    public String getEmployee(int id){
      return employeeRepo.get(id);
    }
}
{% endhighlight %}
### Eclipse Web service generator . 
Even eclipse provide support to create the classes automatically based on WSDL. 
![eclipse](/assets/2019-06-12-SOAP in Java world_1.jpg)

## Bottom up Approach
In bottom up we create a class with some methods and then create service out of it.

### Java approach. 
Use @WebService, @WebMethod and @WebResult annotation to annotate the class and methods. It will somewhat same as generated class file from java or eclipse tool. Once you publish the service wsdl will be available on the URL.  
*http://localhost/EmployeeService?wsdl*

### Eclipse approach. 
Create a simple bean with methods which needs to be expose. Then in eclipse select bottom up approach and select the bean. It will create helper classes and WSDL automatically. 
![eclipse](/assets/2019-06-12-SOAP in Java world_1.jpg)

## Creating client. 
**Eclipse tool** - In both the top down and bottom up approach you have option to create client as well. It creats separate project and creates client classes.
**Java command line tool** - Use following class to create client classes automatically.
*wsimport -keep -p com.xyz.jaxws.client http://localhost/employeeService?wsdl*  
{% highlight Java %}
public class EmployeeServiceClient {
    public static void main(String[] args) throws Exception {
        URL url = new URL("http://localhost/employeeService?wsdl");
 
        EmployeeService employeeService 
          = new EmployeeService(url);
        EmployeeService employeeServiceProxy 
          = employeeService.getEmployeeServiceImplPort();
 
        String name 
          = employeeServiceProxy.getEmployees(1);
    }
}
{% endhighlight  %}

### Publishing WebService 
In the end you would need to publish the web service and make it available. 
{% highlight Java %}
Endpoint.publish(
          "http://localhost:8080/employeeService", 
           new EmployeeServiceImpl());
{% endhighlight %}

