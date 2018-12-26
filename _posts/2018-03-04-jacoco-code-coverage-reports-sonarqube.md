---
layout: post
title: JaCoCo Code Coverage And Reports With Sonarqube
date: 2018-03-04 03:06:05.000000000 -05:00
type: post
parent_id: '0'
published: true
password: ''
status: publish
categories: []
tags: []
redirect_from:
  - /2018/03/04/jacoco-code-coverage-reports-sonarqube/
  - /2018/03/04/jacoco-code-coverage-reports-sonarqube/amp/
cardImage: untitle-2.jpeg
meta:
  _edit_last: '1'
  _thumbnail_id: '271'
  _wp_page_template: default
  post_grid_post_settings: a:10:{s:9:"post_skin";s:4:"flat";s:19:"custom_thumb_source";s:91:"http://abyte.stream/wp-content/plugins/post-grid/assets/frontend/css/images/placeholder.png";s:17:"font_awesome_icon";s:0:"";s:23:"font_awesome_icon_color";s:7:"#737272";s:22:"font_awesome_icon_size";s:4:"50px";s:17:"custom_youtube_id";s:0:"";s:15:"custom_vimeo_id";s:0:"";s:21:"custom_dailymotion_id";s:0:"";s:14:"custom_mp3_url";s:0:"";s:20:"custom_soundcloud_id";s:0:"";}
  _the_champ_meta: a:5:{s:7:"sharing";i:0;s:16:"vertical_sharing";i:0;s:7:"counter";i:0;s:16:"vertical_counter";i:0;s:11:"fb_comments";i:0;}
  _yoast_wpseo_content_score: '60'
  _yoast_wpseo_primary_category: ''
  ampforwp_custom_content_editor: ''
  ampforwp_custom_content_editor_checkbox: ''
  ampforwp-amp-on-off: default
  _yoast_wpseo_focuskw_text_input: JaCoCo Code Coverage And Reports With Sonarqube,
    maven goals to generate reports,
  _yoast_wpseo_focuskw: JaCoCo Code Coverage And Reports With Sonarqube, maven goals
    to generate reports,
  _yoast_wpseo_linkdex: '59'
  _yoast_wpseo_metakeywords: JaCoCo code coverage blog, JaCoCo Code Coverage And Reports
    With Sonarqube, JaCoCo, code coverage, JaCoCo report with Sonarqube, Setting up
    JaCoCo, Importing reports to SonarQube, standalone JaCoCo java agent,
  _yoast_wpseo_metadesc: We have already covered how to do code quality check with
    Sonarqube in my previous blog. In this blog, we will be discussing how can we
    setup JaCoCo a code coverage tool and exports reports to SonarQube. Though report
    can be directly read but having reported at a single place is a good idea. So,
    let get into 'JaCoCo C
author:
  login: sanjaypatel2525
  email: sanjaypatel2525@gmail.com
  display_name: sanjaypatel2525
  first_name: ''
  last_name: ''
---
We have already covered how to do <a href="http://abyte.stream/2018/02/22/code-quality-analysis-sonar/">code quality check with Sonarqube</a> in my previous blog. In this blog, we will be discussing how can we setup JaCoCo a code coverage tool and exports reports to SonarQube. Though report can be directly read but having reported at a single place is a good idea. So, let get into  'JaCoCo Code Coverage And Reports With Sonarqube'
## What this blog will cover.

What is JaCoCo and why you need this?
Setting up JaCoCo.
Importing reports to SonarQube

## What is JaCoCo and why you need this?
<img class="alignnone size-full wp-image-274 " src="{{ site.baseurl }}/assets/img_5a9b6290e8d26.png" alt="" />
JaCoCo is an abbreviation for Java Code Coverage.  It's free to open source library developed by EclEmma. This tool helps us in finding out the piece of code which is not used at all while running the system, It can be manual or by some sort automatic test. When we are working in the team and we have many minds handling the same project there can cases somebody has written a code which doesn't execute at all in runtime or worst is not covered by unit test. Attaching JaCoCo to your unit test make sure you haven't missed out any unwanted untested code. It gives you how many times your code got executed.
## Setting up JaCoCo.
There are multiple ways of running JaCoCo for your project, You can pick any of them depending on your need. Don't worry, in the end, all will generate same data which we can use in Sonarqube.  We are going to export the generated data in *.exec format. There are other formats as well like HTML, zip XML and CSV.

* Install it on Eclipse.
* Install using maven.
* Install standalone JaCoCo java agent on any running java process or server.

## Install it on Eclipse.
This is the easiest approach and best suited when you are developing your code and running unit test cases on the local system.
Steps to install it on eclipse.


### Click or drag below link in your eclipse.
<a title="Drag to your running Eclipse workspace to install EclEmma Java Code Coverage" href="http://marketplace.eclipse.org/marketplace-client-intro?mpc_install=264">
<img style="vertical-align: text-bottom; border: 0px solid black;" src="{{ site.baseurl }}/assets/btn-install.png" alt="Drag to your running Eclipse workspace to install EclEmma Java Code Coverage" />
</a>

### Install from update sites.
Go to Help-> Install new software and put in below URL.
http://update.eclemma.org<img class="alignnone size-full wp-image-269" src="{{ site.baseurl }}/assets/img_5a9b149e3295e.png" alt="" />

### Download plugin zip and install manually.
<a href="http://www.eclemma.org/download.html" rel="nofollow">http://www.eclemma.org/download.html</a>

## Install using maven.
<img class="alignnone size-full wp-image-273 " src="{{ site.baseurl }}/assets/img_5a9b625756041.png" alt="" />
If you are already using maven with your project and want to generate reports using simple maven commands then you can go with this installation.
Steps to install JaCoCo and run reports.


### Open up your pom.xml and add following lines.
{% highlight xml %}
 <groupId>org.jacoco</groupId>
 <artifactId>jacoco-maven-plugin</artifactId>
 <version>0.8.1-SNAPSHOT</version>
</plugin>


 <reporting>
 
 
 <groupId>org.jacoco</groupId>
 <artifactId>jacoco-maven-plugin</artifactId>
 <reportSets>
 <reportSet>
 <reports>
 <report>report</report>
 </reports>
 </reportSet>
 </reportSets>
 </plugin>
 </plugins>
 </reporting>
</project>
{% endhighlight %}


### Run the maven goals to generate reports.
For more options visit.
mvn jacoco:report
For more options visit.
<a href="http://www.eclemma.org/jacoco/trunk/doc/maven.html" rel="nofollow noopener">http://www.eclemma.org/jacoco/trunk/doc/maven.html</a>

### Install standalone JaCoCo java agent on any running java process or server.
This suite best when you do want to run JaCoCo on your live or test environments directly. Also in case, you do not have test setups or local workspace setup. There is standalone JaCoCo jar which you can directly plugin in your remote Java process using command line arguments. Once the java agent is attached to your server process you can open up a port to fetch reports or you can directly go to your server and ask for report dump. This report dump can be in any format CSV, XML, *.exec, HTML or zip. In order to use it with Sonarqube, we have export the report in *.exec format.

Here are the commands you to attach JaCoCo to a running Java process.
-javaagent:C:\dev\servers\jacoco-agent.jar=destfile=< path to jacoco C:\lib\>jacoco.exec,append=true,includes=some.application.*

This Java argument will attach the JaCoCo agent. Once you have enabled the JaCoCo agent you need to run tests manually or automated. This will create jacoco.exec on given location but dump data only on server stop. To avoid server interruption for the report you need to enable command interface using following option.
output=&lt;cpserver&gt;,address=*,<em>port=<some port 6300></em>
## Importing reports to SonarQube
{% highlight properties %}# Generate sonar issues report in html and console
sonar.issuesReport.html.enable=true
sonar.issuesReport.console.enable=true

# This name depends on the configuration in pom.xml. In this example we have ${project.build.directory}/coverage-reports/jacoco-ut.exec entry in our pom.xml
sonar.java.coveragePlugin=jacoco
sonar.jacoco.reportPath=D:/jacocoreport/jacoco.exec
sonar.dynamicAnalysis=reuseReports
sonar.jacoco.reportMissing.force.zero=true
{% endhighlight %}
I ran it on one of my old projects. It looks something like this and it looks like this. For further detail, you can click on coverage percentage.<img class="alignnone size-full wp-image-270" src="{{ site.baseurl }}/assets/img_5a9b5a7882daa.png" alt="" />
