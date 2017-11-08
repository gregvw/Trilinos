#!/usr/bin/env python

import re
import sys
import glob

def repl_dynamic_cast(matchobj):
    g = matchobj.group(1)
    return 'dynamic_cast<'+g.strip()+'&>'

def repl_rcp(matchobj):
    g = matchobj.group(1)

    q = re.match(r" *new(.*)\((.*)\) *", g)
    if (q != None):
        repl = "ROL::makeShared<" + q.group(1).strip() + ">(" + q.group(2) + ")"
        return repl

    q = re.match(r" *new(.*) *", g)
    if (q != None):
        repl = "ROL::makeShared<" + q.group(1).strip() + ">()"
        return repl

    return g

filename = sys.argv[1]

filenames_h = glob.glob(filename + "/**/*.hpp", recursive=True)
filenames_c = glob.glob(filename + "/**/*.cpp", recursive=True)
filenames = filenames_h + filenames_c

writeback = True

for filename in filenames:
    print ("=============================================================================")
    print (filename)
    print ("=============================================================================")

    d = open(filename)
    data = d.read()
    d.close()

    # Remove "using" statements
    data = re.sub('using Teuchos::RCP;', '', data)
    data = re.sub('using Teuchos::rcp;', '', data)
    data = re.sub('using Teuchos::dyn_cast;', '', data)

    # Simple replacements
    data = re.sub('"Teuchos_RCP.hpp"', '"ROL_SharedPointer.hpp"', data)
    data = re.sub('Teuchos::null', 'ROL::nullPointer', data)
    data = re.sub('Teuchos::RCP', 'ROL::SharedPointer', data)
    data = re.sub('RCP', 'ROL::SharedPointer', data)
    data = re.sub('Teuchos::rcp_dynamic_cast', 'ROL::dynamicPointerCast', data)
    data = re.sub('Teuchos::rcp_const_cast', 'ROL::constPointerCast', data)

    # More complex replacement (with make_shared)
    data = re.sub('Teuchos::rcp\((.*)\)', repl_rcp, data)
    data = re.sub('rcp\((.*)\)', repl_rcp, data)
    data = re.sub('Teuchos::dyn_cast<(.*)>', repl_dynamic_cast, data)
    data = re.sub('dyn_cast<(.*)>', repl_dynamic_cast, data)

    print(data)

    if (writeback):
        d = open(filename, "w")
        d.write(data)
        d.close()
