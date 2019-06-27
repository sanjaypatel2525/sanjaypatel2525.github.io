---
layout: post
title: Fine-tune WordPress for Google F1 Micro
date: 2018-02-17 18:12:49.000000000 -05:00
type: post
parent_id: '0'
published: true
password: ''
status: publish
categories: []
tags: []
redirect_from:
  - /2018/02/17/fine-tune-wordpress-google-f1-micro/
  - /2018/02/17/fine-tune-wordpress-google-f1-micro/amp/
cardImage: Untitled-4.jpg
meta:
  _edit_last: '1'
  _thumbnail_id: '161'
  _wp_page_template: default
  post_grid_post_settings: a:10:{s:9:"post_skin";s:4:"flat";s:19:"custom_thumb_source";s:91:"http://abyte.stream/wp-content/plugins/post-grid/assets/frontend/css/images/placeholder.png";s:17:"font_awesome_icon";s:0:"";s:23:"font_awesome_icon_color";s:7:"#737272";s:22:"font_awesome_icon_size";s:4:"50px";s:17:"custom_youtube_id";s:0:"";s:15:"custom_vimeo_id";s:0:"";s:21:"custom_dailymotion_id";s:0:"";s:14:"custom_mp3_url";s:0:"";s:20:"custom_soundcloud_id";s:0:"";}
  _the_champ_meta: a:5:{s:7:"sharing";i:0;s:16:"vertical_sharing";i:0;s:7:"counter";i:0;s:16:"vertical_counter";i:0;s:11:"fb_comments";i:0;}
  _yoast_wpseo_content_score: '90'
  _yoast_wpseo_primary_category: ''
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
In our previous blog, we have discussed <a href="http://abyte.stream/2018/02/17/setting-domain-name-wordpress-site-cloudflare-dns-provider/">setting up domain for your Wordpress website</a>. In continuation to that, in this blog, we will be discussing fine tuning your WordPress Setup to run Google F1 Micro VM Instance.

Google F1 Micro instance gives 30 GB of disk space,  0.6 GB of Memory and 1 CPU, which is no doubt very small configuration.
Wordpress has vast plugin library for plugins. In this blog, we are going to cover few must-have plugins for us to increase reduce our website load and load you website in seconds.

Here are the main things which we are going to focus on.
1. Enabling caching on your WordPress site.
2. Enabling AMP pages for mobile support.
3. Enabling CDN for your website.

# Enabling Caching On Your WordPress Website - WP Super Cache
<img class="irc_mi" src="{{ site.baseurl }}/assets/banner-772x250.png" alt="Image result for wp super cache" width="772" height="264" />
&nbsp;
### 1. Login to your WordPress admin panel
### 2. Go to Plugins from left navigation click Add New button.
<img class="alignnone size-full wp-image-144" src="{{ site.baseurl }}/assets/img_5a886976d328b.png" alt="" />
### 3. Search for WP - Super Cache and install. Once it is installed click on Activate.
### 4. Go to WP - Super Cache settings.
<img class="alignnone size-full wp-image-146" src="{{ site.baseurl }}/assets/img_5a886a11f0be8.png" alt="" />
### 5. Click on Caching on and update status.
<img class="alignnone size-full wp-image-147" src="{{ site.baseurl }}/assets/img_5a886a2daed59.png" alt="" />
This is it your default caching is on. So what we have achieved here is, We remove the need for WordPress to run the code and get a blog for you. It will maintain a superfast cache and will serve the page to you. There is a small cache, In case you don't see your blog updated on your website, just clear the cache with following option.
### 6. Go to content and click on Delete Cache
<img class="alignnone size-full wp-image-148" src="{{ site.baseurl }}/assets/img_5a886ade9cb68.png" alt="" />
### 7. Please use following settings in from advance tab. Other fields you can keep as original.
<img class="alignnone size-full wp-image-149" src="{{ site.baseurl }}/assets/img_5a886b4055693.png" alt="" />
<img class="alignnone size-full wp-image-150" src="{{ site.baseurl }}/assets/img_5a886b5abac95.png" alt="" />
<img class="alignnone size-full wp-image-151" src="{{ site.baseurl }}/assets/img_5a886b7762f71.png" alt="" />
That's all, your site is all cached

# Enabling AMP pages for mobile support - Accelerated Mobile Pages
<img class="transparent" src="{{ site.baseurl }}/assets/banner-772x250.png" alt="https://ps.w.org/accelerated-mobile-pages/assets/banner-772x250.png" />
&nbsp;
This is a very important feature which all site should have, It optimizes your WordPress page or Post to for mobile devices. The HTML is trimmed and optimized in such a way that it feels like the site is loaded in no time.
Google SEO recommends to have AMP pages enabled for your site and this helps your page in terms of SEO and chances of your site to come on top result increases.
### 1. Go to plugins, install and activate Accelerated Mobile Pages.
### 2. There is nothing else now by default your pages and post have  another version, just append '/amp' at the end of your blogs URL

# Enabling CDN for your website - CloudFlare.
<img class="irc_mi" src="{{ site.baseurl }}/assets/Cloudflare-HTTPS-WAF-update.jpg" alt="Image result for cloudflare" width="486" height="243" />
If you are following my complete series of Wordpress on Google Cloud you might have already finished this task. But if not you can always refer my blog to <a href="http://abyte.stream/2018/02/17/setting-domain-name-wordpress-site-cloudflare-dns-provider/">enabling Cloudflare for your WordPress Site</a>.

The main advantage of this site is all your static content like image, HTML, CSS, JS is already cached by one of the best CDN in the world. That meant any request to your site will be served from nearest Cloudflare CDN server.  This definitely reduces your WordPress page load time and relieves your server from extra load.
&nbsp;
