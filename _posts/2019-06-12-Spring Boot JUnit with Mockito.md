---
layout: post
title: Spring Boot JUnit with Mockito. 
date: 2018-12-11 18:57:06.000000000 -05:00
type: post
parent_id: '0'
published: true
password: ''
status: publish
categories: []
tags: []
randomImage: '13'
meta:
  _edit_last: '1'
author:
  login: sanjaypatel2525
  email: sanjaypatel2525@gmail.com
  display_name: sanjaypatel2525
  first_name: ''
  last_name: ''
---
There are cases when you want to write test cases for your method, but the method itslef usage some other method or service, in these scenarios we try to skip the inner methods call or service by mocking it up. 

Mockito is very famous mocking api. With this you can mock any object and just focus on testing of your method only. 

Let's quickly jump into Spring boot JUnit configuration. Add following lines in pom.xml
{% highlight XML %}
<dependency>
	<groupId>org.springframework.boot</groupId>
	<artifactId>spring-boot-starter-web</artifactId>
</dependency>
<dependency>
	<groupId>org.springframework.boot</groupId>
	<artifactId>spring-boot-starter-test</artifactId>
	<scope>test</scope>
</dependency>
{% endhighlight  %}
Once you add boot starter, it will automatically add following dependenies to your project.
* Basic JUnit. 
* Spring unit test framework
* Asserts - AssertJ, Hamcrest Assert, JSONAssert, Java assert
* Mockito. 

### Small example.
{% highlight Java %}
package com.xyz.dao;

public class EmployeeDao {
    ...
	public Employee getEmployee(int id) {
        ....
	}
}

.....

package com.xyz.service;

public class EmployeeBOImpl {
    @Autowire public EmployeeDao employeeDao;
    ...
	public boolean isManager(int id) {
        Employee emp = employeeDao.getEmployee(2);
        return emp.getRole().equals("MANAGER");
	}
}

....

public class EmployeeBOImplTest{
    public EmployeeBOImpl employeeBOImpl;

    ...
    @Before
	public void before() ...

	@After
	public void after() ...

	@BeforeClass
	public static void beforeClass()...

	@AfterClass
	public static void afterClass()...
    ...

    //this method will faile with null pointer exception as employeeDao is not defined for this object.
    @Test
    void testEmployeeBad(){
        employeeBOImpl = new EmployeeBOImpl();
        assertTrue(employeeBOImpl.isManager()); 
    }

    //Use mockito to mock employeeDao
    @Test
    void testEmployeeGood(){
        Employee emp = new Employee();
        emp.setRole("MANAGER");
        
        EmployeeDao employeeDao = mock(EmployeeDao.class);
		when(employeeDao.getEmployee(anyInt())).thenReturn(emp);

        employeeBOImpl = new EmployeeBOImpl();
        assertTrue(employeeBOImpl.isManager()); 
    }
}

{% endhighlight  %}
## Common Mockito function and usage. 
**Using Mockito annotation** In order to make mockito annotations work you would to use one of the below methods. 
* Init Mockito first -- MockitoAnnotations.initMocks(this)
* Or use Mockito runner -- @RunWith(MockitoJUnitRunner.class)  
### doReturn+when 
Eaerlier we have sued when with then return, 'when+thenReturn' has some limitation such it can't work void method also, if you don't use mocked object in 'when' it might call function on actual object and throw some errors. While doreturn+when handles both the scenrios well. It doesn't call actuall method of object but it mockt the function call.  
*doReturn(emp).when(emp).getEmployee(anyInt());*  
### doThrow+when
Same as doRetrun, doThrow can throw a exception on specific method of the object.
*doThrow(NullPointerException.class).when(emp).getEmployee(anyInt());*  
### Add a Spy
Mockito provide a utility a object which keeps an eye on the target object. It tracks the intractions on target object.
{% highlight JAVA %}
Employee emp = spy(new Employee());
emp.setRole("MANAGER");  // the real setRole method is called. 
Mockito.verify(emp).setRole("MANAGER")  // It return true, as setRole was already called on this object.
when(emp.getRole()).thenReturn("MANAGER"); // real method will called here and can throw exception. so use doReturn in place.  
{% endhighlight %}
### Call real object
 In a class you can mock all the methods, but in case you want to call a real method on the class. 
*when(emp.getId()).thenCallRealMethod(); // will call real method rather than mocked one.*
### @InjectMock
 Inject mock takes a mock object and inserts into another object just like autowiring.

### Mock vs Spy.
Spy is a wrapper around actual instance and tracks the activity on instance while mock is barebone mocked instance created for the class. Use spy when you want to track activity on object it works on actuall object, use mock when you just want barebone object of the class. 
EmployeeDao employeeDao = mock(EmployeeDao.class); // create a mocked object of the class
EmployeeDao employeeDao = spy(new EmployeeDao()));  //creates new actual object and wraps a spy

### ArgumentMatcher
You might have seen *anyInt(), anyString()* passed in argument of method when using 'when' or doReturn feature of mockito. These are ArgumentMatcher, you dont need to pass actual values but can match with a argument type or so. For custome class you can use *any(CustomClass.class)*. There is option to create customer matcher as well. 
{% highlight Java %}
public class CustomMatcher implements ArgumentMatcher<CustomClass> {
    private CustomClass customClass;
    // constructors
    @Override
    public boolean matches(customClass customClass) {
        // do something and retun if matches.
    }
} 
{% endhighlight  %}
