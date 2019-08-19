---
layout: post
title: Spring JDBC in Short
date: 2018-12-11 18:57:06.000000000 -05:00
type: post
parent_id: '0'
published: true
password: ''
status: publish
categories: [Spring]
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
First of what all flavours you have from Spring to connect and query DB,
* Spring JDBC
* Spring Data JDBC
* Spring Data JPA  
Spring Data JDBC use Spring JDBC and Spring Data JPA(Relational DB) use Spring Data JDBC internally. They all different libraries provided by Spring. **Spring JDBC** provides you helper classes and utilities to query database and transform it to java object or other way around. **Spring Data JDBC** is another abstraction over Spring JDBC but with CrudRepostiroy facility, with this you don't need to write crud query for your entities. **Spring Data JPA** is another abstraction over Spring Data JDBC but with persistent api support. You can do ORM here which takes care of creating, quering, saving the objects directly into DB. Also it gives more control over session and object lifecycle. 

In this blog we will be discussing about common functionality provided by Spring JDBC.

## Configuring JDBC Datasource
{% highlight JAVA %}
import org.springframework.jdbc.datasource.DriverManagerDataSource;
import javax.sql.DataSource;
@Configuration
public class AppConfiguration {
    @Bean
    public DataSource getDataSource() {
        DriverManagerDataSource dataSource = new DriverManagerDataSource();
        dataSource.setDriverClassName("com.mysql.jdbc.Driver");
        dataSource.setUrl("jdbc:mysql://localhost:3306/sampledb");
        dataSource.setUsername("username");
        dataSource.setPassword("password");
        return dataSource;
    }
}
{% endhighlight  %}

## JdbcTemplate to execute queries. 
Once you have configured the datasource you can autowire the JDBCTemplate in your application and use to execute queries and map the data back to java object. This class give many functions for your need but here are some important ones. 
* Update
Update take insert or update queries and update the record and return the number of records inserted/updated.
Update has many different overloaded variant one of them is as below. 
{% highlight JAVA %}
jdbcTemplate.update("INSERT INTO EMPLOYEES VALUES (?, ?, ?, ?)", 5, "Jack", "Daniels", "USA"); 
{% endhighlight  %}
* execute
Execute also has many variants, execute with void returns are mainly used for DDL statements. Other execute methods can be used for normal queries and stored procedures. 
{% highlight JAVA %}
jdbcTemplate.execute("create table mytable (id integer, name varchar(100))");
{% endhighlight  %}
* queryForObject
queryForObject takes a select query can return a list or single object. queryForObject has many different overloaded variant one of them is as below. 
{% highlight JAVA %}
jdbcTemplate.queryForObject("select count(*) from employees", Integer.class);

Employees employees = jdbcTemplate.queryForObject("select * from employees where emp_id=?",10, 
      new EmployeesMapper());      // use for single

List<Employees> employees = jdbcTemplate.queryForObject("select * from employees", 
      new EmployeesMapper()); // use for list

private static final class EmployeesMapper implements RowMapper<Employees>() {
    public Employees mapRow(ResultSet rs, int rowNum) throws SQLException {
        Employees employees = new Employees();
        employees.setFirstName(rs.getString("first_name"));
        employees.setLastName(rs.getString("last_name"));
        return employees;
    }
}
{% endhighlight  %}

## NamedParameterJdbcTemplate to execute queries.
NameParameter you can name the parameter for value mapping with ':' operator and put the value in map. 
{% highlight JAVA %}
Map<String, String> namedParameters = Collections.singletonMap("first_name", firstName);
namedParameterJdbcTemplate.queryForObject("select * from employees where first_name=:first_name", namedParameters, new EmployeesMapper());
{% endhighlight  %}

## Using SimpleJdbc classes
There are two classes provided by Spring, SimpleJdbcInsert and SimpleJdbcCall. These classes helps programmer to skip insert queries, procedure calls and write all the queries in form of Java classes. 
For example, Here is an insert. 
{% highlight JAVA %}
SimpleJdbcInsert insertEmployee = new SimpleJdbcInsert(dataSource).withTableName("Employee");
Map<String, Object> parameters = new HashMap<String, Object>(3);
parameters.put("id", employee.getId());
parameters.put("first_name", employee.getFirstName());
parameters.put("last_name", employee.getLastName());
insertEmployee.execute(parameters);
{% endhighlight  %}

