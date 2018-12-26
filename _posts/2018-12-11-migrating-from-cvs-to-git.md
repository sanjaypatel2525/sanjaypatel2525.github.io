---
layout: post
title: Migrating from CVS to Git.
date: 2018-12-11 18:57:06.000000000 -05:00
type: post
parent_id: '0'
published: true
password: ''
status: publish
categories: []
tags: []
redirect_from:
  - /2018/12/11/migrating-from-cvs-to-git/
  - /2018/12/11/migrating-from-cvs-to-git/amp/
cardImage: Untitled.jpg
meta:
  _edit_last: '1'
  _wp_page_template: default
  _adinserter_block_exceptions: ''
  post_grid_post_settings: a:10:{s:9:"post_skin";s:4:"flat";s:19:"custom_thumb_source";s:92:"https://abyte.stream/wp-content/plugins/post-grid/assets/frontend/css/images/placeholder.png";s:17:"font_awesome_icon";s:0:"";s:23:"font_awesome_icon_color";s:7:"#737272";s:22:"font_awesome_icon_size";s:4:"50px";s:17:"custom_youtube_id";s:0:"";s:15:"custom_vimeo_id";s:0:"";s:21:"custom_dailymotion_id";s:0:"";s:14:"custom_mp3_url";s:0:"";s:20:"custom_soundcloud_id";s:0:"";}
  _the_champ_meta: a:5:{s:7:"sharing";i:0;s:16:"vertical_sharing";i:0;s:7:"counter";i:0;s:16:"vertical_counter";i:0;s:11:"fb_comments";i:0;}
  _yoast_wpseo_content_score: '90'
  _yoast_wpseo_primary_category: ''
  ampforwp_custom_content_editor: ''
  ampforwp_custom_content_editor_checkbox: ''
  ampforwp-amp-on-off: default
  _thumbnail_id: '366'
author:
  login: sanjaypatel2525
  email: sanjaypatel2525@gmail.com
  display_name: sanjaypatel2525
  first_name: ''
  last_name: ''
---
In good old days, CVS used to be one of the most used SCM (source control management) tool. With the growing technology and the need of distributed systems, Git was born and with this CVS started rusting out. This is much of history for now.
There are cases when we have a requirement to move our codebase from CVS to Git without losing commit histories. If we dontIn this blog, we are going to discuss the same.
### Basic requirement.

* Need Python installed.
* Access to CVS server for getting the full code base.
* CVSNT server setup. (This is not required if you are already on CVS server. CVS client doesn't work)
* Need Git installed.

## Steps to convert CVS to Git
### Install Python 2.7.x
Download and install Python 2.7.x and set it to OS environment path.
### Install CVSNT server.  (Not required when you are already on CVS Server)
Download and install CVSNT server version. You will need it only once and then you can uninstall it.
<a href="https://www.march-hare.com/cvspro/">https://www.march-hare.com/cvspro/</a>

### Download the cvs2git tool and extract it to a new folder 'temp'.
cvs2git comes prepackaged with cvs2svn tool. You can download it from the following location.<br />
<a href="http://cvs2svn.tigris.org/servlets/ProjectDocumentList?folderID=2976&amp;expandFolder=2976&amp;folderID=0">http://cvs2svn.tigris.org/servlets/ProjectDocumentList?folderID=2976&amp;expandFolder=2976&amp;folderID=0</a>

### Go to CVS  server and create a zip of the full code base.
Find out the CVSRoot and zip the parent folder.
### Extract the zipped codebase to the folder 'temp'.
### Open the command prompt and navigate to the following directory, replace the version with the correct version.
temp/cvs2svn-<strong>x.x.x
</strong>
### Run the following command from the command prompt.
python.exe cvs2git --blobfile=git-blob.dat --dumpfile=git-dump.dat --use-external-blob-generator --username=cvs2git C:/temp/&lt;Extracted CVSFolder&gt;<br />
Once this command has completed successfully it will create two files, git-blob.dat and git-dump.dat. This command will fail if you have CVS client version installed.

### Initialize an empty bare repository in temp folder.
git init --bare &lt;YourProrjectName&gt;.git

### Navigate to &lt;YourProrjectName&gt;.git
cd &lt;YourProrjectName&gt;.git

### Import dat files in repository.
git fast-import --export-marks=../../cvs2svn-2.5.0/git-marks.dat <../../cvs2svn-2.5.0/git-blob.dat<br />
git fast-import --import-marks=../../cvs2svn-2.5.0/git-marks.dat <../../cvs2svn-2.5.0/git-dump.dat<br />
By now your local Git project is ready.

### Call garbage collector. (Optional)
git gc --prune=now

### You can verify source history using the below command.
gitk<br />
It opens up the git history browser.
### Here onwards, the steps are for pushing the codebase to a common remote Git server.


### Clone your bare repository in 'temp'.
Navigate to temp and run this following command. The following command will create the folder named &lt;YourProrjectName&gt;.<br />
git clone &lt;YourProrjectName&gt;.git

### Navigate to &lt;YourProrjectName&gt; and add remote location.
Remote location will be you common git server location which will be shared by your team.<br />
git remote add origin remote repository URL

### Push your local git to the remote.
git push origin master

### Now you can delete your bare repository 'temp/&lt;YourProrjectName&gt;.git'

That's all about CVS to git codebase migration. Next topic we are going to update the ant build script and configure it will git setup.