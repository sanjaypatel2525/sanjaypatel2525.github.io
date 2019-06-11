---
layout: post
title: Hibernate in Short
date: 2018-12-11 18:57:06.000000000 -05:00
type: post
parent_id: '0'
published: true
password: ''
status: publish
categories: []
tags: []
randomImage: '15'
meta:
  _edit_last: '1'
author:
  login: sanjaypatel2525
  email: sanjaypatel2525@gmail.com
  display_name: sanjaypatel2525
  first_name: ''
  last_name: ''
---
Hibernate, hibernate-jpa, Spring ORM, Spring data JPA and many more. All this seems to be confusing but they all are just jars which are  automatically added when you add the below line you pom.xml.
{% highlight XML %}
<dependency>
  <groupId>org.springframework.boot</groupId>
  <artifactId>spring-boot-starter-jpa</artifactId>
</dependency> 
{% endhighlight %}
Basically, Spring boot gives a starter project for your data acces layer which is based on ORM concept and by default usage hibernate as persistant api. This blog we are not going to discuss about how spring encapsulated hibernate under it but we are going to discuss about Hibernate basics and there Spring alternative way of writing code. 
![gras](/assets/2019-06-04-Hibernate in Short.png)
### Hibernate in action with JPA.
* Create Entity class
* Create Repository interface
* Use repository.
{% highlight JAVA %}
@Entity
@Table(name = "company") // Optional as it take class name
@Document // incase of nosql DB
public class Employee{
  
  @Id
  @Column(name = "id") // Optional asit take field name
  @GeneratedValue // Automatically generated values, kind of autoincrement in table.
  Long id;
  String firstname;
  String lastName;
}

....
@Repository
public interface EmployeeRepository extends JpaRepository<Employee, Long> {
    public List<Employee> findByLastName(String lastName); // select * from Employee where lastname=:lastname
}

....
// Using repository.
Employee emp = employeeRepository.save(emp);
List<Employee> emps = employeeRepository.findByLastName("Clark");
employeeRepository.delete(emp);

{% endhighlight %}


### Pros and Cons of Hibernate over Plain JDBC.
Hibernate is a ORM **Object-relational mapping**, It maps the entity classes to database tables. Sometime it easy to map one to one entity to table but sometime it really gets complex when you have inheritance, different kind of table relationships such as one to one, one to many and  manay to many. Also it really gets messy if you don't handle the fetch and cascade paramters properly. But over all it has many advantages over its disadvantages. That why it is so popular.
* Reduces and reusage code.  
You dont need write query again and again for insert update or anything. 
* Independent of database.
You can use any underlying database without modifying your code. You just need change dialect and pom entry to update driver of the database.   
*spring.jpa.properties.hibernate.dialect =  org.hibernate.dialect.DB2Dialect*
{% highlight JAVA %}
<dependency>
  <groupId>com.ibm.db2.jcc</groupId>
  <artifactId>db2jcc4</artifactId>
</dependency>
{% endhighlight %}
* Lazy loading.  
Hibernate you can set the property to lazy load the related table data and hibernate takes care of fetching the data when the data is required.
* Transaction management.  
Hibernate maintains the transaction and revert all the changes in case if there is any exception.
* Caching.  
In memory is always faster then the reaching out to DB for the data. Hibernate has inmemory caching option it keeps data in memory which is faster to access and is very smart when to commit and when not to. 
* Connection Pooling.  
Inbuilt support of connection pooling. It create pool of connection which is faster then creating a new connection and terminating it. 
* Supports HQL(superset of JPQL).  
HQL is a query language supported in Hibernates, In queries you can direclty use classname and it autmatically genrates the database queries. 
* Automatic versioning.  
Maintains different version of data and saves the headach of missing the data. 
* Perfomance  
The query generated here will be highly optimized in most of the cases. 


### Setting up Datasource. 
If you start with starter-jpa datasource and started inmemory DB jar such as h2database etc, you don't need to define the datasource but you can define and override the default ones. Here is Java way of defining it.
{% highlight JAVA %}
@Bean
public DataSource dataSource() {
    DriverManagerDataSource dataSource = new DriverManagerDataSource();
 
    dataSource.setDriverClassName("com.mysql.cj.jdbc.Driver");
    dataSource.setUsername("admin");
    dataSource.setPassword("password");
    dataSource.setUrl(
      "jdbc:mysql://localhost:3306/testDB?createDatabaseIfNotExist=true"); 
     
    return dataSource;
}
{% endhighlight  %}
Here is defining the datasource by just property files. Make sure you use exact keys. 
{% highlight Property %}
spring.datasource.driver-class-name=com.mysql.cj.jdbc.Driver
spring.datasource.username=admin
spring.datasource.password=password
spring.datasource.url=jdbc:mysql://localhost:3306/testDB?createDatabaseIfNotExist=true
{% endhighlight %}

### Object States in Hibernate
* Transient  
Object just created but not attached to hibernate session and not connected to database. 
* Persistent  
Object attached hibernate session and connected to database, any update to this will be relected in database. 
* Detached  
Object detached from session. It happend when session is closed or you can manually detach it.

### Session, SessionFactory and Transaction. 
SessionFactory provide you the session class which can be used to attach, detach, save, query etc. Session supports transaction as well.
{% highlight Java %}
Session session = factory.openSession();
Transaction tx = null;

try {
   tx = session.beginTransaction();
   // do some work
   ...
  session.update(emp);
   ...
   tx.commit();
}

catch (Exception e) {
   if (tx!=null) tx.rollback();
   e.printStackTrace(); 
} finally {
   session.close();
}
{% endhighlight %}

### Annotations used class table mapping.
**@Entity** - Makes the class available for Table mapping.  
**@Table** - It is optional. Used with @Entity, in case you want different name of the table.  
**@Column** - It is optional, in case you want different name of the field.  
**@Id** - Makes field primary key, you will have to use @GeneratedValue if you want generated values of id.  
**@GeneratedValue** - To generate field value. There four type of it. Auto(Unique value accorss database), identity(Unique accors class hierarchy), sequence(custome sequence ex:start:100, increment- 10), Table.  
**@Version** - For versioning and concurrency you can keep a field which defines a version.  
**@Transient** - In case you dont want a field to be part of database mapping.  

{% highlight JAVA %}
@Entity
@Table(name = "Employee_Custome_Table_Name")
public class Employee implements Serializable {
  @Id
  @GeneratedValue
  private int id;

  private String firstName; // column name defaults to first_name
  
  @Column(name = "custom_last_name") // column name is specified,  custom_last_name
  private String lastName; 

  @Version
  private Date lastUpdate;

  @Transient
  private String sessionId; // will not be stored in db
}
{% endhighlight  %}

## Mapping Tables.
**@OnetoOne** - When there is one to one relation, but again it has four different types.  
**@PrimaryKeyJoinColumn** - When two table share same primary key. Such as Employee and EmployeeDetails. 
{% highlight JAVA %}
@Entity
@Table(name = "employee")
public class Employee implements Serializable {
   
  @Id
  @Column(name = "id")
  @GeneratedValue
  private int id;
   
  @OneToOne(cascade = CascadeType.MERGE)
  @PrimaryKeyJoinColumn
  private EmployeeDetail employeeDetail;
   
  ...
}
 
@Entity
@Table(name = "employee_detail")
public class EmployeeDetail implements Serializable {
 
  @Id
  @Column(name = "id")
  private int id;
   
  ...
}
{% endhighlight  %}

**@MapsId** - When tables joined by foreign and primary key of two tables and both class cotains object of each other. Such Employee and EmployeeBadge.

{% highlight JAVA %}
@Entity
@Table(name = "employee")
public class Employee implements Serializable {
 
  @Id
  @Column(name = "id")
  @GeneratedValue
  private int id;
   
  @OneToOne
  @MapsId
  @JoinColumn(name = "employee_badge_id")
  private EmployeeBadge employeeBadge;
   
  ...
}
 
@Entity
@Table(name = "employee_badge")
public class EmployeeBadge implements Serializable {
 
  @Id
  @Column(name = "ID")
  @GeneratedValue
  private Integer id;
 
  @OneToOne(mappedBy = "employee", cascade = CascadeType.ALL)
  private Employee employee;
 
  ....
}
{% endhighlight  %}

**@OneToMany & @ManyToOne** - Both are represent same logical thing. A Project can have many employeed which is one to many. Many employee can be mapped a single project which is many to one. It depends how you organize your java class. 
{% highlight JAVA %}
// One to Many example. 
@Entity
@Table(name = "project")
public class Project{
  @Id
  @GeneratedValue
  public Long projectId;

  @OneToMany(mappedBy = "project")
  @JoinColumn(name = "projectId")
  List<Employee> employees;
  ....
}
@Entity
@Table(name = "employee")
public class Employee{
  @Id
  @GeneratedValue
  public Long employeeId;
  ....
}

// Many to one example

@Entity
@Table(name = "project")
public class Project{
  @Id
  @GeneratedValue
  public Long projectId;
  ....
}
@Entity
@Table(name = "employee")
public class Employee{
  @ManyToOne
  @JoinColumn(name = "projectId") // if this annotaition not present it defaults to table relation @JoinTable(name = "project_employee")
  Project project;
  .....
}

// all of the above was unidirectional relation using a foreign key in Employee as projectid. We can do the same relation with third relation table.
{% endhighlight  %}


**@ManyToMany** - Many to many relation always needs a third table. For example employee can work on many tasks and a task can have multiple employees working on it. 
{% highlight JAVA %}
@Entity
@Table(name = "task")
public class Task{
  @Id
  @GeneratedValue
  public Long taskId;

  @ManyToMany(mappedBy = "tasks")
  List<Employee> employees
  ....
}
@Entity
@Table(name = "employee")
public class Employee{
  @ManyToMany(cascade = { CascadeType.ALL })
  @JoinTable(
      name = "Employee_Project", 
      joinColumns = { @JoinColumn(name = "employee_id") }, 
      inverseJoinColumns = { @JoinColumn(name = "task_id") }
  )
  List<Task> tasks;

  @Id
  @GeneratedValue
  public Long employeeId;
  .....
}
{% endhighlight  %}

### HQL Query.
Hibernates provides many way to query you table. Native, HQL, criteria etc. HQL is extension of SQL almost same as SQL standards but it usage class name in place of table name. Also, binging a value is very easy. You can retrieve query object from session and Query interface has following methods. 
* public int executeUpdate()
* public List list()
* public Query setFirstResult(int rowno) //For pagination set the first row num
* public Query setMaxResult(int rowno) // Page count
* public Query setParameter(int position, Object value)
* public Query setParameter(String name, Object value)

{% highlight JAVA %}
String hql = "FROM Employee E WHERE E.id = :employee_id";
Query query = session.createQuery(hql);
query.setParameter("employee_id",1000);
List results = query.list();

....

String hql = "UPDATE Employee set age = :age "  + 
             "WHERE id = :empId";
Query query = session.createQuery(hql);
query.setParameter("age", 25);
query.setParameter("empId", 100);
int count = query.executeUpdate();
System.out.println("Rows affected: " + count);
{% endhighlight %}

### Criteria API.
Criteria api helps you to ommit sqls or HQL where you have to pass the query as string. Criteria API provides you set of classes and utilities with which you can form queries. You can create Criteria object form session.

Criteria ct = session.createCriteria(Employee.class);
List<Employee> results = ct.list();

**Using Restriction** - Restriction class with and, or, eq, ne, gt, lt, ge, le, like,ilike, between, in, conjuction, disjunction, isNull, isNotNull, sqlRestriction(String sql, Object value, Type type) etc. 

{% highlight JAVA %}
Criteria ct = session.createCriteria(Employee.class);
ct.add(Restrictions.lt("age",20));
ct.add(Restrictions.sqlRestriction("{alias}.firstname like 'San%'"));
ct.setFirstResult(1);
ct.setMaxResults(20);
ct.addOrder(Order.desc("age"));
List<Employee> results = ct.list()
{% endhighlight  %}

**Using Projection to get selecte Columns**

{% highlight JAVA %}
Criteria ct = session.createCriteria(Employee.class);
ProjectionList projList = Projections.projectionList();
projList.add(Projections.property("firstName"));
projList.add(Projections.property("lastName"));
ct.setProjection(projList);
List<Object[]> results = ct.list();

Criteria ct = session.createCriteria(Employee.class);
ct.setProjection(Projections.rowCount());
List<Long> results = ct.list();
{% endhighlight %}

* avg(String propertyName)
* count(String propertyName)
* countDistinct(String propertyName
* max(String propertyName)
* min(String propertyName)
* sum(String propertyName)

**Using query by example**
Query by example usage a filter object. Let's say you want filter a employee based on salary and age. Create a filter object with salary and age set. 
{% highlight JAVA %}
Employee e = new Employee();
e.setAge(20);
e.setSalary(1000);
Criteria ct = session.createCriteria(Employee.class);
ct.add(Example.create(e));
List<Employee> results = ct.list();
{% endhighlight  %}

### Native queries.
{% highlight JAVA %}
String sql = "SELECT * FROM EMPLOYEE WHERE id = :employee_id";
SQLQuery query = session.createSQLQuery(sql);
query.addEntity(Employee.class);
query.setParameter("employee_id", 10);
List<Employee> results = query.list();
{% endhighlight  %}

## Caching in Hibernate.
Hibernates has two level level of cache. Firstlevel is by default there and maintained by session container. You can't change first level cache. Second level cache is left for developer where he can defined his own caching strategies at each class level. Such as Employee class secondary cache can use any of the below strategies.  
**Transactional** - Cached data will be retained only in transaction.  
**Read-write** - It create read write lock on table and updates if the data has been updated.  
**Nonstrict-read-write** - When data rarely changes and you don't bother about stale data.
**Read-only** - When data never changes.  