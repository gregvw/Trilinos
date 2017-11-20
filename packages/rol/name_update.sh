cname() {
  out='s/'$1'/'$2'/g'
  echo $out
  grep -rl $1 ./ | xargs sed -i $out 

}

cname 'ROL::SharedPointer' 'ROL::SharedPointer'
cname 'ROL::makeShared' 'ROL::makeShared'
cname '"ROL_SharedPointer.hpp"' '"ROL_SharedPointer.hpp"'
cname 'ROL::constPointerCast' 'ROL::constPointerCast'
cname 'ROL::dynamicPointerCast' 'ROL::dynamicPointerCast'
cname 'ROL::staticPointerCast' 'ROL::staticPointerCast'
cname 'ROL::nullPointer' 'ROL::nullPointer'

