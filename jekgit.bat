@echo off

IF [%1] == [] (
  echo "Please enter a git commit message"
  exit
)
 
jekyll build &&   cd _site &&   git add . &&   git commit -am "%1" &&   git push origin master &&   cd .. &&   echo "Successfully built and pushed to GitHub."