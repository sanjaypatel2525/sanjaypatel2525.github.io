---
layout: post
title: Spring Security Simple.
date: 2018-12-11 18:57:06.000000000 -05:00
type: post
parent_id: '0'
published: true
password: ''
status: publish
categories: []
tags: []
randomImage: '2'
meta:
  _edit_last: '1'
author:
  login: sanjaypatel2525
  email: sanjaypatel2525@gmail.com
  display_name: sanjaypatel2525
  first_name: ''
  last_name: ''
---
This blog we are going to discuss a easiest approach to implement spring security, Spring Security Details is will more into customization points wrt to standard spring architecture. 

Spring Security focuses on two main points.
* Authentication : Verify/Identify the requester's/user's idenity.
* Authorization: Allow/block resource access based on requester/user's access level.

There are many authentication/authorization models supported in Spring such, LDAP, oauth, oauth2, OpenId, HTTP Basic etc. Also, Spring security has many feature Single such as sign-on, LDAP. Remember me etc.

## Spring XML based Configuration. 
### POM needs following depedencies. 
{% highlight XML %}
<dependency>
    <groupId>org.springframework</groupId>
    <artifactId>spring-webmvc</artifactId>
</dependency>
<dependency>
    <groupId>org.springframework.security</groupId>
    <artifactId>spring-security-web</artifactId>
</dependency>
<dependency>
    <groupId>org.springframework.security</groupId>
    <artifactId>spring-security-core</artifactId>
</dependency>
<dependency>
    <groupId>org.springframework.security</groupId>
    <artifactId>spring-security-config</artifactId>
</dependency>  
{% endhighlight %}

### web.xml
{% highlight XML %}
<filter>
    <filter-name>springSecurityFilterChain</filter-name>
    <filter-class>org.springframework.web.filter.DelegatingFilterProxy</filter-class>
</filter>
<filter-mapping>
    <filter-name>springSecurityFilterChain</filter-name>
    <url-pattern>/*</url-pattern>
</filter-mapping>
<context-param>
    <param-name>contextConfigLocation</param-name>
    <param-value>  
            /WEB-INF/spring-servlet.xml  
            /WEB-INF/spring-security.xml  
        </param-value>
</context-param>  
{% endhighlight %}

### spring-security.xml
Basic HTTP authentication. 
{% highlight XML %}
<beans:beans xmlns="http://www.springframework.org/schema/security"  
    xmlns:beans="http://www.springframework.org/schema/beans"  
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"  
    xsi:schemaLocation="http://www.springframework.org/schema/beans  
    http://www.springframework.org/schema/beans/spring-beans.xsd  
    http://www.springframework.org/schema/security  
    http://www.springframework.org/schema/security/spring-security.xsd">  
    <http auto-config="true">  
        <intercept-url pattern="/admin" access="hasRole('ROLE_ADMIN')" />  
    </http>  
    <authentication-manager>  
      <authentication-provider>  
        <user-service>  
        <user name="admin" password="password" authorities="hasRole(ROLE_ADMIN)" />  
        </user-service>  
      </authentication-provider>  
    </authentication-manager>  
</beans:beans>  
{% endhighlight %}

## Java based configuration.
### pom.xml
{% highlight XML %}
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-security</artifactId>
</dependency>
{% endhighlight %}
If add following dependency, authetnication is by default enabled, in case you want disable auto configuration use @SpringBootApplication(exclude = { SecurityAutoConfiguration.class }) at main class.

### Create configuration class.
{% highlight JAVA %}
@Configuration
@EnableWebSecurity
public class BasicConfiguration extends WebSecurityConfigurerAdapter {
 
    @Override
    protected void configure(AuthenticationManagerBuilder auth)
      throws Exception {
        auth
          .inMemoryAuthentication()
          .withUser("user1")
            .password("password1")
            .roles("USER")
            .and()
          .withUser("admin")
            .password("password")
            .roles("USER", "ADMIN");
    }
 
    @Override
    protected void configure(HttpSecurity http) throws Exception {
        http
          .authorizeRequests()
          .anyRequest()
          .authenticated()
          .and()
          .httpBasic();
    }
}
{% endhighlight %}

Using the above code you have implmeneted basic spring security with inmemeory hardcoded username and passwords. In case you want to use database for user name and password. We will not be discussing user registration here. Following are the important component of database base user authentication.
* Database connection
* PasswordEncoder - Helps to encode password before saving to DB and matching at the time of authentication.
* User from DB (jdbcAuthentication and DaoAuthenticationProvider/UserDetailService)
* User Roles

### Here is example of jdbcAuthentication.
{% highlight JAVA %}

@Configuration
public class WebMvcConfig implements WebMvcConfigurer {
    @Autowired
    private BCryptPasswordEncoder bCryptPasswordEncoder; //Passwordencoder

    @Autowired
    private DataSource dataSource; //Datasource you need configure using property files.
    @Override

    protected void configure(AuthenticationManagerBuilder auth)
          throws Exception {
      auth.
              jdbcAuthentication()
              .usersByUsernameQuery("select email, password, active from user where email=?")
              .authoritiesByUsernameQuery("select u.email, r.role from user u inner join user_role ur on(u.user_id=ur.user_id) inner join role r on(ur.role_id=r.role_id) where u.email=?")
              .dataSource(dataSource)
              .passwordEncoder(bCryptPasswordEncoder);
    }
    ....
}
{% endhighlight %}

### Here is example of DaoAuthenticationProvider/UserDetailService.
{% highlight JAVA %}

@Configuration
public class WebMvcConfig implements WebMvcConfigurer {

    @Autowired
    private BCryptPasswordEncoder bCryptPasswordEncoder; //Passwordencoder
    @Autowired
    private CustomUserDetailsService userDetailsService;
    
    @Override
    protected void configure(AuthenticationManagerBuilder auth)
      throws Exception {
        auth.authenticationProvider(authenticationProvider());
    }
    
    @Bean
    public DaoAuthenticationProvider authenticationProvider() {
        DaoAuthenticationProvider authProvider
          = new DaoAuthenticationProvider();
        authProvider.setUserDetailsService(userDetailsService);
        authProvider.setPasswordEncoder(encoder());
        return authProvider;
    }
    .....
}

...
@Service
public class CustomUserDetailsService implements UserDetailsService {
 
    @Autowired
    private UserRepository userRepository;
 
    @Override
    public UserDetails loadUserByUsername(String username) {
        User user = userRepository.findByUsername(username);
        if (user == null) {
            throw new UsernameNotFoundException(username);
        }
        return new CustomUserPrincipal(user);
    }
}

.....
public class CustomUserPrincipal implements UserDetails {
    private User user;
 
    public CustomUserPrincipal(User user) {
        this.user = user;
    }
    //...
}

.........
@Data
@Entity
@Table(name = "user")
public class User {

    @Id
    @GeneratedValue(strategy = GenerationType.AUTO)
    @Column(name = "user_id")
    private int id;
	
    @Column(name = "email")
    private String email;
    
	@Column(name = "password")
    private String password;
    @Column(name = "name")
    
	private String name;
    @Column(name = "last_name")
    private String lastName;
    @Column(name = "active")
    private int active;
    
    @JoinTable(name = "user_role", joinColumns = @JoinColumn(name = "user_id"), inverseJoinColumns = @JoinColumn(name = "role_id"))
    @ManyToMany(cascade = CascadeType.ALL)
	private Set<Role> roles;

}

@Data
@Entity
@Table(name = "role")
public class Role {
    @Id
    @GeneratedValue(strategy = GenerationType.AUTO)
    @Column(name = "role_id")
    private int id;
    @Column(name = "role")
    private String role;
}
{% endhighlight %}

### User Roles
User roles is crucial part when it comes to authorization of resources based on user authority. You can have any custom role defined in role table and assign those roles to user. On Controllers or services you can access the user roles and write your logic based on roles. There are many ways to check roles. 
1. Inject HttpServletRequest arugument and use function request.isUserInRole("ROLE_<CustomRole>").
2. Inject SecurityContextHolderAwareRequestWrapper arugument and use function request.isUserInRole("ROLE_<CustomRole>").
3. Read it from context.
{% highlight JAVA %}
SecurityContext context = SecurityContextHolder.getContext();
  if (context == null)
      return false;

  Authentication authentication = context.getAuthentication();
  if (authentication == null)
      return false;

  for (GrantedAuthority auth : authentication.getAuthorities()) {
      if (role.equals(auth.getAuthority()))
          return true;
}
{% endhighlight %}
4. Use SpEL
{% highlight JAVA %}
@Configuration
@EnableWebSecurity
@EnableGlobalMethodSecurity(prePostEnabled = true)
public class SecurityConfig extends WebSecurityConfigurerAdapter {
    ...
}
@Service
public class CustomService {
    @PreAuthorize("hasRole('ROLE_ADMIN')")
    public List<CustomClass> findAll() { ... }
    ...
}
{% endhighlight %}

5. Use @Secured (This one is old compared to preauthorize )
Preauthorize is comparativley new and supports SpEL such as below.  
{% highlight JAVA %}
......
// usage of SpEL
@PreAuthorize("#address.id == principal.id and hasRole('ROLE_ADMIN')")
public void someMethod(Address address)

.....
// Usage of Secured annotation
@Secured({ "ROLE_ADMIN", "ROLE_USER" })
public void someMethod(Address address)
{% endhighlight %}
6. @Use RolesAllowed (This is same as secured annotaion but it is as per Java standard )