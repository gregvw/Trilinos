# Script for cleaning out the build directory

cwd=`pwd`
cwd=`basename $cwd`
if [ $cwd == "build" ]; then 
  mkdir temp
  cp *.sh temp
  ls | grep -v temp | xargs rm -rf
  cp temp/* ./
  rm -rf temp

  for filename in *.sh; do
    chmod u+x $filename
  done
else
  echo "clean can only be run from the build directory"
fi
