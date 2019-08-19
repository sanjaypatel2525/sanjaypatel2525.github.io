---
layout: post
title: JUnit Basics with JUnit 5. 
date: 2018-12-11 18:57:06.000000000 -05:00
type: post
parent_id: '0'
published: true
password: ''
status: publish
categories: [Spring]
tags: []
randomImage: '16'
meta:
  _edit_last: '1'
author:
  login: sanjaypatel2525
  email: sanjaypatel2525@gmail.com
  display_name: sanjaypatel2525
  first_name: ''
  last_name: ''
---
First of all, what is JUnit? JUnit is Java provided library to automate your unit tests cases. A project can have million lines of code and probably thousand methods, Just to make sure that the methods are behaving same as what they suppose to be, we write unit test cases for those methods. This keeps your project bug safe with future edits on method. 

Let's quickly jump into JUnit configuration. Add following lines in pom.xml
{% highlight XML %}
<dependency>
	<groupId>org.junit.jupiter</groupId>
	<artifactId>junit-jupiter-engine</artifactId>
	<version>${junit.jupiter.version}</version>
</dependency>
{% endhighlight  %}

## Basics of JUnit. 

{% highlight XML %}
<dependency>
    <groupId>org.junit.jupiter</groupId>
    <artifactId>junit-jupiter-engine</artifactId>
</dependency>
<dependency>
    <groupId>org.junit.platform</groupId>
    <artifactId>junit-platform-runner</artifactId>
    <scope>test</scope>
</dependency>
{% endhighlight %}

{% highlight JAVA %}
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
 
import com.xyz.junit5.XYZBOImpl;
 
public class XYZBOImplTest {
    public XYZBOImpl impl;
    @BeforeAll
    static void setup(){
        System.out.println("@BeforeAll will be executed only once when this XYZBOImplTest first test method is called");
        XYZBOImpl impl = XYZBOImpl();
    }
     
    @BeforeEach
    void callBeforEachMethod(){
        System.out.println("@BeforeEach will be executed before each test methods in XYZBOImplTest");
    }
     
    @Tag("dev")
    @Test
    void tesXYZBOImplMethod1()
    {
        System.out.println("Testing Method 1 of XYZBOImpl");
        Assertions.assertEquals( "some return value" , impl.method1());
    }
     
    @Tag("prod")
    @Disabled
    @Test
    void tesXYZBOImplMethod2()
    {
        System.out.println("Testing Method 2 of XYZBOImpl"));
        Assertions.assertThrows(Exception.class, () -> {
            Assertions.assertEquals( "some return value" , impl.method2()));
        });
        
    }
     
    @AfterEach
    void callAfterEachMethod(){
        System.out.println("@AfterEach is execute after each test methods in XYZBOImplTest");
    }
     
    @AfterAll
    static void destroy(){
        System.out.println("@AfterAll will be executed only once when this XYZBOImplTest last test method is called");
    }
}

//Example test only selected packages
@RunWith(JUnitPlatform.class)
@SelectPackages("com.xyz.junit5.examples2")
@IncludePackages("com.xyz.junit5.examples2.includeMe")
@ExcludePackages("com.xyz.junit5.examples2.excludeMe")
@ExcludeTags("PROD")
public class JUnit5TestSuiteExample1
{
}

// Example two test only selected classes.
@RunWith(JUnitPlatform.class)
@SelectPackages("com.xyz.junit5.examples2")
public class JUnit5TestSuiteExample1
{
}
{% endhighlight %}

### Common annotations.
**@Test** - Used to decorate method which will considered a test case.   
**@ParameterizedTest** - Run a test method multiple times with different parameter.
@ParameterizedTest
@ValueSource(strings = { "racecar", "radar", "ssoss" })
void palindromes(String candidate) {
    assertTrue(StringUtils.isPalindrome(candidate));
}
**@TestMethodOrder and @Order(n)** - TestMethodOrder is class decorator and makes the class test run in a order, @Order is a method decorator and tell the order of the method by numeric values.
**@BeforeEach and @AfterEach** - Methods decorator, makes methods to be called before and after of each method.  
**@BeforeAll and @AfterAll** - Methods decorator, makes methods to be called before and after of each test class.
**@Tag** - Helps to tag method and later helps to filter the test methods with the use of runner. You can see the example above.
**@Disabled** - Disable the class or method in the test runner.

### Assertions
Assertion checks the conditions and throws assert exception which in the end fails the Junit test case. If none of the asserts in the test case throws assert exception that test case will be marked pass.
* assertSame(expected, actual,"fail msg") // evaluate expected!=actual
* assertNotSame(expected, actual,"fail msg") //evaluate expected==actual
* assertEquals(expected, actual,"fail msg") // evaluate expected.equals(actual)
* assertArrayEquals(['a','b','c'],['a','b','c'],"fail msg")
* assertTrue(method1(),"fail msg")  // will pass if method return true.
* assertFalse(method1(),"fail msg")  // will pass if method return false.
* assertNull(obj,"fail msg")  // will pass if obj is null
* assertNotNull(obj,"fail msg")  // will pass if obj is not null
* fail("fail msg")  // Throw assert error with msg.

JUnit 5 has "fail msg" as last parameter, It accepts a supplier interface as well for lazy initialization. 
* AssertAll // for grouping multiple asserts and combining message.
* assertIterableEquals //Will match each record of two iterable(ArrayList, linked list etc)

### Assumption
Assumption helps in writing conditional Junit test case. Test cases can be skipped if assumption is not matched. If test case assumption is not met, test case will be aborted and skipped.
* Assumptions.assumeTrue("DEV".equals(System.getProperty("ENV")),"msg")
* Assumptions.assumeTrue("DEV".equals(System.getProperty("ENV")),"msg")

JUnit 5 comprises of three modules.
* JUnit Platform
* JUnit Jupiter
* JUnit Vintage