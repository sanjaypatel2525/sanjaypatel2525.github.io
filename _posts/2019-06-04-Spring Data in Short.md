---
layout: post
title: Spring Data in Short
date: 2018-12-11 18:57:06.000000000 -05:00
type: post
parent_id: '0'
published: true
password: ''
status: publish
categories: [Spring]
tags: []
randomImage: '14'
meta:
  _edit_last: '1'
author:
  login: sanjaypatel2525
  email: sanjaypatel2525@gmail.com
  display_name: sanjaypatel2525
  first_name: ''
  last_name: ''
---
There are many different data storage system in the market currently and in order to connect and query different system you might have to use different query syntax. Spring Data is one of the advance module and has many sub-module to support different DBs, but in the end the main Spring Data defines common abstraction layer. This abstraction defines a common standard for different system and helps to reuse the same code at different places.

For example if you have to query for Employee object present in MongoDB and MySQL. You will have to use following two queries respectively.   
select * from Employee where id = 5;  
db.employee.find( { _id: 5 } );  
But with spring data you just need to define one method and Spring data takes care of forming internal queries.  
Employee findById(int id);  

## Configuring repository.
Add Spring data in POM configuration.
{% highlight XML %}
<dependency>
   <groupId>org.springframework.data</groupId>
   <artifactId>spring-data-jpa</artifactId>
</dependency>
{% endhighlight %}  
### Enable reporsitory package scan.
Enable package scan by @EnableJpaRepositories(basePackages = "com.mycompany.repository") or with @SpringBootApplication.

### Create repository interface.
Implement any of the repository interface. 
{% highlight JAVA %}
@Repository
public interface EmployeeRepository extends JpaRepository<Employee, Long> {
    public List<Employee> findByLastName(String lastName); // select * from Employee where lastname=:lastname
}
// Create Employee entity class.
@Entity
@Document // incase of nosql DB
class Employee{
  Long id;
  String firstname;
  String lastName;
}
{% endhighlight %}

There are three different repositories Spring Data Supports. 
* CrudRepository
* PagingAndSortingRepository
* JpaRepository

### CrudRepository
CrudRepository supports basic crud operation such as create, reade, update delete. 
{% highlight JAVA %}
public interface CrudRepository<T, ID extends Serializable>
  extends Repository<T, ID> {
 
    <S extends T> S save(S entity);
 
    T findOne(ID primaryKey);
 
    Iterable<T> findAll();
 
    Long count();
 
    void delete(T entity);
 
    boolean exists(ID primaryKey);
}

//Example
public interface EmployeeRepository extends CrudRepository<Employee, Long> {
    public List<Employee> findByLastName(String lastName); // select * from Employee where lastname=:lastname
}

....
employeeRepository.findOne(id); // select * from Employee where id=:id
employeeRepository.delete(employee); // delete from Employee where id=:employee.id

{% endhighlight %}  

### PagingAndSortingRepository
PagingAndSortingRepository implements CrudRepository so you get all the perks of CrudRepository and something extra. 
{% highlight JAVA %}
public interface PagingAndSortingRepository<T, ID extends Serializable> 
  extends CrudRepository<T, ID> {
 
    Iterable<T> findAll(Sort sort);
 
    Page<T> findAll(Pageable pageable);
}
.....
//Examples
public interface EmployeeRepository extends CrudRepository<Employee, Long> {
    public List<Employee> findByLastName(String lastName, Pageable pageable); // select * from Employee where lastname=:lastname
}
......
Sort sort = new Sort(new Sort.Order(Direction.DESC, "lastName"));
Pageable pageable = new PageRequest(0, 15, sort);
employeeRepository.findByLastName("Jack", pageable); // select * from Employee where lastname=:lastname and rownum>0 and rownum<15  order by lastName desc;
{% endhighlight %}  
PagingAndSortingRepository gives you feature to get records in pages and sort the results. 

### JpaRepository
{% highlight JAVA %}
JpaRepository implements PagingAndSortingRepository and gives you some persistent level feature.
public interface JpaRepository<T, ID extends Serializable> extends
  PagingAndSortingRepository<T, ID> {
 
    List<T> findAll();
 
    List<T> findAll(Sort sort);
 
    List<T> save(Iterable<? extends T> entities);
 
    void flush();
 
    T saveAndFlush(T entity);
 
    void deleteInBatch(Iterable<T> entities);
}
{% endhighlight %}

## Spring Data features. 
Let us see some benefits of having spring data. 
### No DAO interfaces. 
You don't need to write your DAO interface if you can manage with basic crud features such as insert, delete, update, read etc.
### Support custom queries.
In case you need to cocnifgure custom query which are not supported by Crud interface, then you can define your custom implementation class where you can use custom queries. 
{% highlight JAVA %}
//Examples
public interface EmployeeRepositoryCustom{
   public List<Employee> runComplexQuery();
}

public interface EmployeeRepository extends CrudRepository<Employee, Long>,EmployeeRepositoryCustom {
    public List<Employee> findByLastName(String lastName, Pageable pageable); // select * from Employee where lastname=:lastname
}

public class EmployeeRepositoryCustomImpl implements EmployeeRepositoryCustom{
   public List<Employee> runComplexQuery(){
     ....
     // Use any kind of query generation classes or just right plain query and use jdbctemplate.
   };
}

{% endhighlight %}

### Query based on method name.
Based on method names spring tries to create query for them, for example.
Employee findByFirstNameAndLastName(String firstname, String lastName); // where firstname=:firstname and lastnmae= : lastname
Employee findByAgeGreaterThanOrderByFirstnameDesc(int age); // where age>:age order by firstname desc

### @Query annotation.
You can use @Query annotation to query by native sql queries or JPQL. For example.
{% highlight JAVA %}
@Query("from Employee where lastName=:lastName")
List<Employee> findByLastName(@Param("firstName") String lastName)
//or
@Query("from Employee where lastName=?1")
List<Employee> findByLastName(String lastName)
// Or
@Query(value ="select * from EMPLOYEE_TABLE where lastName=:lastName", nativeQuery = true)
List<Employee> findByLastName(@Param("firstName") String lastName)

//insert or update queries has to be marked as @modified
 @Query("UPDATE Employee SET firstName = :firstName ")
@Modifying
void setFirstName(@Param("firstName") String firstName);
{% endhighlight %}