---
layout: post
title: EJB in Java. 
date: 2018-12-11 18:57:06.000000000 -05:00
type: post
parent_id: '0'
published: true
password: ''
status: unpublished
categories: [J2EE]
tags: []
randomImage: '18'
meta:
  _edit_last: '1'
author:
  login: sanjaypatel2525
  email: sanjaypatel2525@gmail.com
  display_name: sanjaypatel2525
  first_name: ''
  last_name: ''
---
EJB are wrapper around any kind of component, if you need data access, directory, resources, services, JMS, RMI, SOA etc. EJB Provides a standard way to wrap these thing under common umbrella which has many features. It can be compared to POJO but with lot of advance feature which you dont need write.
* Reusing the code. 
* Scalability and reliability, Do pooling, limit connection and access easily, deploy on cluster environment. 
* Concurrency management, Easily configure concurrency using small annotation over methods such @Lock(READ), @Lock(WRITE). 
* Transaction handling, You don't need to do any changes transaction is handled easily in EJB just annotate with @TransactionAttribute(modes), it covers JDBC, JPA, JCA and JMS easily. 
* Dependency injection. EJB gives easy dependency injection mechanism, such JNDI stored, JDBC connection, JMS, JCA, JTA, JPA, EJB itself. 
{% highlight JAVA %}
@Stateless
public class MyAccountsBean {

    @EJB SomeOtherBeanClass someOtherBean;
    @Resource UserTransaction jtaTx;
    @PersistenceContext(unitName="AccountsPU") EntityManager em;
    @Resource QueueConnectionFactory accountsJMSfactory;
    @Resource Queue accountPaymentDestinationQueue;

    public List<Account> processAccounts(DepartmentId id) {
        // Use all of above instance variables with no additional setup.
        // They automatically partake in a (server coordinated) JTA transaction
    }
}
@EJB MyAccountsBean accountsBean;    
{% endhighlight %}

* Smart interaction, Easy transition with JPA, if entity is stateless it will create new transaction and isolate the JPA changes. In case of sharing Make is @Stateful and extend the scope by @PersistentContent(unitName="AccountsPU, type=EXTENDED). So it caches the value from JPA and serve the same across multiple calls on same transaction.
* Lifecycle hooks, it provides lifecycle hooks to inject code for initializing, destroying and cleaning activity. 
* Security based access, With simple annotation methods and classes can be secured. It provides user and roles access to methods in case they are required RBAC(Role based access control).
* Standardization and portability

EJB sounds very nice starting from EJB 3.0, EJB 2.0 Entity bean has moved to separate Java EE specification JPA entities. EJB mainly had 3 type of bean. 
* Session bean - Contains business logic and have two type Stateful and stateless. 
* Message Driven Bean - It encapsulate onMessage methods and get invoked whenever there is a message received from queue or topic. 
* Entity Bean - It was used to persist the state of object but it has moved to JPA now. EJB 3.0 supports JPA it has removed the entity bean concept. It had two type CMP and BMP, CMP container provides many inbuilt function such persist, connect, transaction etc, while BMP programmer has to write that. 

### Stateless
It has following methods
* @Stateless
* @PostConstruct
* @PreDestroy
Example of Stateful using JNDI.
{% highlight JAVA %}
package com.javatpoint;  
import javax.ejb.Remote;  
  
@Remote  
public interface AdderImplRemote {  
int add(int a,int b);  
}  

package com.javatpoint;  
import javax.ejb.Stateless;  
  
@Stateless(mappedName="st1")  
public class AdderImpl implements AdderImplRemote {  
  public int add(int a,int b){  
      return a+b;  
  }  
}  

package com.javatpoint;  
import javax.naming.Context;  
import javax.naming.InitialContext;  
  
public class Test {  
public static void main(String[] args)throws Exception {  
    Context context=new InitialContext();  
    AdderImplRemote remote=(AdderImplRemote)context.lookup("st1");  
    System.out.println(remote.add(32,32));  
}  
}  
{% endhighlight  %}

### Stateful
It has following methods.
* @Stateful
* @PostConstruct
* @PreDestroy
* @PrePassivate
* @PostActivate

{% highlight JAVA %}
package com.javatpoint;  
import javax.ejb.Remote;  
@Remote  
public interface BankRemote {  
    boolean withdraw(int amount);  
    void deposit(int amount);  
    int getBalance();  
}  

package com.javatpoint;  
import javax.ejb.Stateful;  
@Stateful(mappedName = "stateful123")  
public class Bank implements BankRemote {  
    private int amount=0;  
    public boolean withdraw(int amount){  
        if(amount<=this.amount){  
            this.amount-=amount;  
            return true;  
        }else{  
            return false;  
        }  
    }  
    public void deposit(int amount){  
        this.amount+=amount;  
    }  
    public int getBalance(){  
        return amount;  
    }  
}  

package com.javatpoint;  
import java.io.IOException;  
import javax.ejb.EJB;  
import javax.naming.InitialContext;  
import javax.servlet.ServletException;  
import javax.servlet.annotation.WebServlet;  
import javax.servlet.http.HttpServlet;  
import javax.servlet.http.HttpServletRequest;  
import javax.servlet.http.HttpServletResponse;  
@WebServlet("/OpenAccount")  
public class OpenAccount extends HttpServlet {  
    //@EJB(mappedName="stateful123")  
    //BankRemote b;  
    protected void doGet(HttpServletRequest request, HttpServletResponse response)  
                throws ServletException, IOException {  
        try{  
            InitialContext context=new InitialContext();  
            BankRemote b=(BankRemote)context.lookup("stateful123");  
              
            request.getSession().setAttribute("remote",b);  
            request.getRequestDispatcher("/operation.jsp").forward(request, response);  
          
        }catch(Exception e){System.out.println(e);}  
    }  
}  

{% endhighlight  %}
