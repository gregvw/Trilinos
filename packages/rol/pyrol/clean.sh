# Script for cleaning out the build directory

cwd=`pwd`
cwd=`basename $cwd`
if [ $cwd == "build" ]; then 
  mkdir temp
  mv "do-configure-pyrol.sh" temp
  mv "clean.sh" temp
  ls | grep -v temp | xargs rm -rf
  mv temp/* ./
  rm -rf temp
  chmod u+x "do-configure-pyrol.sh"
  chmod u+x "clean.sh"
else
  echo "clean can only be run from the build directory"
fi
