---
title: Websphere Commerce build with Git
layout: post
date: '2017-12-11 21:16:05 -0500'
type: post
published: true
status: publish
cardImage: Untitled-2.jpg
redirect_from:
  - /2018/12/11/websphere-commerce-build-with-git/
  - /2018/12/11/websphere-commerce-build-with-git/amp/
categories:
- Websphere Commerce
meta:
  _edit_last: 1
  _thumbnail_id: 373
  _wp_page_template: default
  _adinserter_block_exceptions: ''
  post_grid_post_settings: a:10:{s:9:"post_skin";s:4:"flat";s:19:"custom_thumb_source";s:92:"https://abyte.stream/wp-content/plugins/post-grid/assets/frontend/css/images/placeholder.png";s:17:"font_awesome_icon";s:0:"";s:23:"font_awesome_icon_color";s:7:"#737272";s:22:"font_awesome_icon_size";s:4:"50px";s:17:"custom_youtube_id";s:0:"";s:15:"custom_vimeo_id";s:0:"";s:21:"custom_dailymotion_id";s:0:"";s:14:"custom_mp3_url";s:0:"";s:20:"custom_soundcloud_id";s:0:"";}
  _the_champ_meta: a:5:{s:7:"sharing";i:0;s:16:"vertical_sharing";i:0;s:7:"counter";i:0;s:16:"vertical_counter";i:0;s:11:"fb_comments";i:0;}
  _yoast_wpseo_content_score: 60
  _yoast_wpseo_primary_category: 3
  ampforwp_custom_content_editor: ''
  ampforwp_custom_content_editor_checkbox: ''
  ampforwp-amp-on-off: default
author:
  login: sanjaypatel2525
  email: sanjaypatel2525@gmail.com
  display_name: sanjaypatel2525
  first_name: ''
  last_name: ''
---

Websphere Commerce (WCS) is e-Commerce software platform. It supports multiple business models to support a wide range of customer requirements.
WCS did not have Git support till version 8.  It has introduced Git support from version 9 onwards. Git is a free open source distributed source control management software.
This blog we are going to discuss Git configuration with WCS build.
### Prerequisite:
- Git needs to be installed on the build system

### Copy2 the template file from wcbd template to wcbd  folder and update the names as given below.

![gras]({{ site.baseurl }}/assets/img_5c1021100f5c3.png)
- extract-git.private.properties<br />
- extract-git.properties<br />
- extract-git.xml


### Configure build.property to pick new extract-git.xml

extract.ant.file=${basedir}/extract-git.xml
### Configure git URL and default branch or tag name in extract-git.properties.
{% highlight properties %}
#-------------------------------------------------------------------------------
# git.url
#
# The git url variable. On Windows, use the escaped backslash (\) as the path
# separator if repository is local.
#-------------------------------------------------------------------------------
git.url=yourgitip:yourgit:port/yourprojectname.git

#-------------------------------------------------------------------------------
# git.branchOrTag
#
# The branch or tag of the package/module to check out.
#-------------------------------------------------------------------------------
git.branchOrTag=development
{% endhighlight %}

###  Configure git username and password in extract-git.private.properties
### Run the build and test it using the following command.
{% highlight properties %}
git.password=<someusername>
git.username=<somepassword>
{% endhighlight %}
### Update the extract-git.xml with below ant code.
{% highlight xml %}
<project name="extract-git" default="all">

  <available property="found.${ant.project.name}.properties"
             file="${basedir}/${ant.project.name}.properties"
             type="file" />
  <fail message="${basedir}/${ant.project.name}.properties does not exist."
        unless="found.${ant.project.name}.properties" />
  <property file="${basedir}/${ant.project.name}.properties" />

  <encodeProperties file="${ant.project.name}.private.properties" />
  <decodeLoadProperties file="${ant.project.name}.private.properties" />

  <!--
   Runs the source extraction process.
  -->
  <target name="all">
    <condition property="branchOrTag" value="${build.branchOrTag}" else="${git.branchOrTag}">
        <isset property="build.branchOrTag"/>
    </condition>
    <if>
      <istrue value="${extract.update.mode}" />
      <then>
        <if>
          <available file="${module.dir}" type="dir" />
          <then>
            <echo message = "GIT Command: git pull --tags" />
            <exec executable = "git" dir = "${module.dir}" failonerror="true">
              <arg value = "pull" />
              <arg value = "--tags" />
            </exec>
            <echo message = "GIT Command: git reset --hard origin/${branchOrTag}" />
            <exec executable = "git" dir = "${module.dir}" failonerror="true">
              <arg value = "reset" />
              <arg value = "--hard" />
              <arg value = "origin/${branchOrTag}" />
            </exec>
          </then>
          <else>
            <echo message = "GIT Command: git clone -b ${branchOrTag} http://${git.username}:${git.password}@${git.url}" />
            <exec executable = "git" dir = "${source.dir}" failonerror="true">
              <arg value = "clone" />
              <arg value = "-b" />
              <arg value = "${branchOrTag}" />
              <arg value = "http://${git.username}:${git.password}@${git.url}" />
            </exec>
          </else>
        </if>
      </then>
      <else>
        <echo message = "GIT Command: git clone -b ${branchOrTag} http://${git.username}:${git.password}@${git.url}" />
        <exec executable = "git" dir = "${source.dir}" failonerror="true">
          <arg value = "clone" />
          <arg value = "-b" />
          <arg value = "${branchOrTag}" />
          <arg value = "http://${git.username}:${git.password}@${git.url}" />
        </exec>
      </else>
    </if>
    <defaultexcludes add="CVSROOT/**" />
  </target>

</project>
{% endhighlight %}
### Run the build and test it using the following command.
./wcbd-ant -buildfile wcbd-build.xml
This is all about configuring Git with WCBD ant script. This script covers build in both the modes full and update. Also, this build supports code checkout form tag and branch both.
&nbsp;