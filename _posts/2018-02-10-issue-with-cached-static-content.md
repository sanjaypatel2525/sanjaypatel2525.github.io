---
layout: post
title: Issue With Cached Static Content - Cache Buster
date: 2018-02-10 17:08:05.000000000 -05:00
type: post
parent_id: '0'
published: true
password: ''
status: publish
categories:
- Websphere Commerce
tags: []
redirect_from:
  - /2018/02/10/issue-with-cached-static-content/
  - /2018/02/10/issue-with-cached-static-content/amp/
cardImage: Untitled-5.jpg
meta:
  _edit_last: '1'
  _yoast_wpseo_primary_category: '3'
  _yoast_wpseo_content_score: '90'
  post_grid_post_settings: a:10:{s:9:"post_skin";s:4:"flat";s:19:"custom_thumb_source";s:88:"http://techys.tk/wp-content/plugins/post-grid/assets/frontend/css/images/placeholder.png";s:17:"font_awesome_icon";s:0:"";s:23:"font_awesome_icon_color";s:7:"#737272";s:22:"font_awesome_icon_size";s:4:"50px";s:17:"custom_youtube_id";s:0:"";s:15:"custom_vimeo_id";s:0:"";s:21:"custom_dailymotion_id";s:0:"";s:14:"custom_mp3_url";s:0:"";s:20:"custom_soundcloud_id";s:0:"";}
  _thumbnail_id: '134'
  _wp_page_template: default
  _yoast_wpseo_focuskw_text_input: Issue with cached static content - cache buster
  _yoast_wpseo_focuskw: Issue with cached static content - cache buster
  _yoast_wpseo_linkdex: '75'
  _yoast_wpseo_metakeywords: cache buster, static file cache, cache expiry, cache
    version
  _yoast_wpseo_metadesc: Understanding the different aproaches to resolve issue cached
    static content. It involve static content versoning, shortning cache expiry and
    automate the process with grunt task runner.
  use_ampforwp_page_builder: 'yes'
  ampforwp_page_builder_enable: ''
  ampforwp_custom_content_editor: ''
  ampforwp_custom_content_editor_checkbox: ''
  ampforwp-amp-on-off: default
  _the_champ_meta: a:5:{s:7:"sharing";i:0;s:16:"vertical_sharing";i:0;s:7:"counter";i:0;s:16:"vertical_counter";i:0;s:11:"fb_comments";i:0;}
author:
  login: sanjaypatel2525
  email: sanjaypatel2525@gmail.com
  display_name: sanjaypatel2525
  first_name: ''
  last_name: ''
---
Many us might have faced issue with cached static content - cache buster, Every deployment whenever there is change in static files it becomes a painful job to ask your end customer to clear the cache. In order to get rid of this problem, here are few ideas for cache buster.

Append version number in front of changed files references ( /path/file.js?v=1.1)
Reduce cache expiry header
Use grunt scripts to handle cache issue.

## Issue with cached static content - cache buster
.
Let's dive into details.
### 1. Appending version number approach
This is the easiest approach to resolve the static content cache issue, But required extra efforts each time you change file. Let's say I have project, something shown as below.
<img class="size-medium wp-image-6 aligncenter" src="{{ site.baseurl }}/assets/exmple2-278x300.jpg" alt="" width="278" height="300" />
In case you have some changes in 1.js, then the references for 1.js has to be update in each files where it is used with '/1.js?v=&lt;new version&gt;', in this both the JSPs.
Pros:
* Very easy to implement.
* Doesn't requires much planning.

Cons:
* Requires updating many file depending on how many places static files is reference.
* Chances of manual errors.

### 2. Reduce cache expiry
Cache expiry headers tells browser when to fetch new version of the file.
<img class="alignnone size-full wp-image-8 " src="{{ site.baseurl }}/assets/img_5a7f228a2f838-768x182.png" alt="" />
Setting this value to 24hrs will fetch new files everyday but since browser will call this file everyday it will increase your page load time also will put some extra load on your servers. This parameter can be configured on both web and app servers.
Pros:
* Easy to configure.

Cons:
* Increase the page load time.
* Increase the server load.

### 3. Use grunt (task runner) while deployment.
It is bit complex and depending how complex is your project structure it can take time to implement. It requires deployment script changes as well.
Pros:

Unlike first one it doesn't require any changes to other files on each deployment
Can be used to minify and compress files as well

Cons:
* One time setup requires some time and depending on how complex is you project structuring it might require lot of changes.

Grunt is a task runner. The idea is to segregate all the CSS and JS files and put them up in single location. On each deployment, the script will minfy and compresses the file present at that location. Then the file is renamed to filename + file checksum, it makes sure if the file changed then the name is changed as well. The last step of script is to update the filename in all the places where it is referenced. For more information please check my other blog <a href="https://blogbeats.me/blog/india/minifying-with-grunt-jenkins/579c82c8e4b05e7c55a2036c" rel="nofollow">link</a>.

Please visit my sites tool section as well.<a href="https://abyte.stream/file-difference/http://techys.tk/file-difference/"> Find file differences</a>
