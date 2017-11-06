#!/usr/bin/env python

import re
import sys
import glob

def repl_rcp(matchobj):
    g = matchobj.group(1)

    q = re.match(r" *new(.*)\((.*)\) *", g)
    if (q != None):
        repl = "std::make_shared<" + q.group(1).strip() + ">(" + q.group(2) + ")"
        return repl

    q = re.match(r" *new(.*) *", g)
    if (q != None):
        repl = "std::make_shared<" + q.group(1).strip() + ">()"
        return repl

    return g

filename = sys.argv[1]

filenames_h = glob.glob(filename + "/**/*.hpp", recursive=True)
filenames_c = glob.glob(filename + "/**/*.cpp", recursive=True)
filenames = filenames_h + filenames_c

writeback = False

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
    data = re.sub('"Teuchos_RCP.hpp"', '<memory>', data)
    data = re.sub('Teuchos::null', 'nullptr', data)
    data = re.sub('Teuchos::RCP', 'std::shared_ptr', data)
    data = re.sub('RCP', 'std::shared_ptr', data)
    data = re.sub('Teuchos::rcp_dynamic_cast', 'std::dynamic_pointer_cast', data)
    data = re.sub('Teuchos::dyn_cast', 'dynamic_cast', data)
    data = re.sub('dyn_cast', 'dynamic_cast', data)
    data = re.sub('Teuchos::rcp_const_cast', 'std::const_pointer_cast', data)

    # More complex replacement (with make_shared)
    data = re.sub('Teuchos::rcp\((.*)\)', repl_rcp, data)
    data = re.sub('rcp\((.*)\)', repl_rcp, data)

    print(data)

    if (writeback):
        d = open(filename, "w")
        d.write(data)
        d.close()
