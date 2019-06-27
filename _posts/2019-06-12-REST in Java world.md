---
layout: post
title: REST Services in Java world. 
date: 2018-12-11 18:57:06.000000000 -05:00
type: post
parent_id: '0'
published: true
password: ''
status: publish
categories: []
tags: []
randomImage: '20'
meta:
  _edit_last: '1'
author:
  login: sanjaypatel2525
  email: sanjaypatel2525@gmail.com
  display_name: sanjaypatel2525
  first_name: ''
  last_name: ''
---
JAX-RS define specification for REST webservice. REST principal goes very well with HTTP protocol that's the reason people normally interpret REST as REST over HTTP service. There are many libraries are there in market which helps in creating REST services Spring Rest, JAX-RS Jersey, Apache CXF, RestEasy. All are quite a bit same and have matching syntax. In this we will discuss mainly about JAX-RS standard and Jersey implementation of JAX-RS, later little bit on Spring REST.

## Spring with Jersey
Spring has it's own rest framework and with different annotations but let's look at JAX-RS standard J2EE JAX-RS annotation with Spring in action. 
{% highlight XML %}
<dependencies>
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-jersey</artifactId>
    </dependency>
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-test</artifactId>
        <scope>test</scope>
    </dependency>
</dependencies>
{% endhighlight  %}
{% highlight Java %}
@Component
public class JerseyConfig extends ResourceConfig
{
    public JerseyConfig()
    {
        register(EmployeeController.class);
    }
}

@Service
@Path("/employee")
public class EmployeeController {
    ...
    @GET
    @Produces(MediaType.APPLICATION_JSON)
    @Path("/{id}")
    public Employee getEmployee(@PathParam("id") int id) {
        return employeeRepo.get(id);
    }

    @POST
    @Consumes(MediaType.APPLICATION_JSON)
    public Employee saveEmployee(Employee emp) {
        return employeeRepo.save(emp);
    }

    // Form custom response object, use context, use formparam.
    @POST
    public Response updateManager(@FormParam("managerID") managerID, @Context UriInfo uriInfo) {
        ...
        emp = emp.setManagerId(managerID)
        emp = employeeRepo.save(emp);
        String urlPath = String.format("%s/%s",uriInfo.getAbsolutePath().toString(); // printing url path just to showcase.
        return Response.status(Response.Status.CREATED.getStatusCode())
          .header(
            "Location", 
            urlPath, 
            employee.getId())).build();
    }

}

@SpringBootApplication
public class Application {

    public static void main(String[] args) {
        SpringApplication.run(Application.class, args);
    }
}
{% endhighlight  %}

### JAX-RS annotation
Check example above to see the annotation usage.  
**@GET** - Method decorator and handle HTTP get request.  
**@Produces & @Consumes** - Defines datatype accepted/produced by method for ex text/xml, text/JSON etc.  
**@Path** - Bind method or class to a URL path and calls it when the path is matched.  
**@PathParam** - Bind the URL matching value to method argument.  
**@QueryParam** - Binds the URL parameter to method argument, Maps GET parameters.  
**@FormParam** - Binds the POST parameters to method arguments.  
**@HeaderParam** - Binds header value to method parameters.  
**@POST** - Binds the complete post parameter to a object in method argument. It usage Java serialization and de-serialization to create Java object.  
**@PUT & @DELETE** - Put is same as post but there is slight difference in logic. DELETE is also same as POST but logically the code inside should do some deletion.  
**@Context** - Getting URI related information.  

## Spring Rest framework.
{% highlight XML %}
<dependency>
  <groupId>org.springframework.boot</groupId>
  <artifactId>spring-boot-starter-web</artifactId>
</dependency>
{% endhighlight  %}
{% highlight Java %}
@RequestMapping("/employee")
public class EmployeeController {
    ...
    @RequestMapping(value="/{id}",method = RequestMethod.GET,produces="application/json")
    // or @GetMapping
    // optional @ResponseStatus(HttpStatus.OK)
    public Employee getEmployee(@PathVariable("id") int id) {
        return employeeRepo.get(id);
    }

    @RequestMapping(value="/{id}",method = RequestMethod.POST,produces="application/json")
    // or @PostMapping
    public Employee saveEmployee(Employee emp) {
        return employeeRepo.save(emp);
    }

    // Form custom response object, header information, use requesparam.
    @POST
    public ResponseEntity<Employee> updateManager( @RequestParam(value = "managerId", required = false, defaultValue = "123")  managerID, @RequestHeader("Accept-Encoding") String encoding, HttpServletRequest request, HttpServletResponse response) { //in case you need request and response variables. 
        
        ...
        emp = emp.setManagerId(managerID)
        emp = employeeRepo.save(emp);
        System.out.println(encoding) // printing encoding path just to showcase.
        HttpHeaders headers = new HttpHeaders();
        headers.add("Custom-Header", "foo");
            
        return new ResponseEntity<>(
          emp, headers, HttpStatus.OK);
    }

}
{% endhighlight %}
### Spring MVC rest related annotations .
**@RestController** - @ResponseBody + @Controller is @RestController, It adds filter to serialize the response.  
**@RequestMapping** - Maps the URL path to class and methods, also specifies the content type of request and response.  Has these fields - method, header, name, value, produces, consumes.  
{% highlight Java %}
@RequestMapping(value = {"/somePath1","/somePath2"}, 
    method = {RequestMethod.GET,RequestMethod.POST},
    consumes = {"application/json","application/xml"},
    produces = { "application/json"},
    headers = {"application/json"})
public String someMethodForTwoPaths() {...}
{% endhighlight  %}
**@RequestParam** - @RequestParam maps the get post value to method argument. You can take default values as well.  
{% highlight Java %}
@GetMapping("/somepath")
public String somepath(@RequestParam("id") String id) { .. }
{% endhighlight  %}
**@PathVariable** - Path variable used to map URL path value to a method argument.
{% highlight Java %}
@GetMapping("/employee/{id}")
public String getEmployee(@PathVariable("id") String id) {...}
{% endhighlight  %}
**@SessionAttribute** - Map session values to method argument.   
{% highlight Java %}
@GetMapping("/getSessionValue")
public String getSessionValue(@SessionAttribute(name = "sessionId") String sessionId) {...}
{% endhighlight  %}
**@CookieValue** - Map cookie value to method argument.   
{% highlight Java %}
@GetMapping("/getCookieValue")
public String getCookieValue(@SessionAttribute(name = "var1") String var1) {...}
{% endhighlight %}