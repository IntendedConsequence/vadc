"""
Copyright Demetri Spanos
MIT/BSD
"""


import os, sys, tempfile, struct, ctypes, re, collections

fields = ["return_t", "name", "arg_ts"]
CBound = collections.namedtuple("CBound", ["return_t","name","arg_ts"])

class Clib(object):
  pass

class scaffolld(object):
  pass

amap = {
  "void*" : ctypes.c_void_p,
  "int" : ctypes.c_int,
  "int32_t" : ctypes.c_int,
  "uint32_t" : ctypes.c_uint,
  "int64_t" : ctypes.c_longlong,
  "uint64_t" : ctypes.c_ulonglong,
  "float" : ctypes.c_float,
  "double" : ctypes.c_double,
  "char*" : ctypes.c_char_p
}

def find_structs (fn):
  patt = re.compile("\n[a-zA-Z][^{;]{1,}struct {[^}]{1,}}[^;]{1,};",re.DOTALL)
  cands = patt.findall(open(fn).read())
  return

def find_funcs (fn):
  patt = re.compile("\n[a-zA-Z][^{;=]{1,}{",re.DOTALL)
  cands = patt.findall(open(fn).read())
  
  res = []
 
  for c in cands:
    ret = c.strip().split()[0]
    fn  = c.strip().split()[1]
    if ret in ["struct", "typedef"]: continue
    if fn in ["struct","typedef"]: continue
    p1 = c.find("(")
    p2 = c.find(")")
    arg_string = c[p1+1:p2].strip()
    if arg_string == "": continue
    args = arg_string.split(",")
    arg_types = []
    for arg in args:
      is_ptr = arg.find("*") > -1
      arg = arg.replace("*"," ").split()[0]
      if is_ptr:
        if arg == "char": arg = "char*"
        else: arg = "void*" 
      arg_types.append(arg)
    ok = True
    if ret not in amap and ret != "void": ok = False
    for atype in arg_types:
      if atype not in amap: ok = False
    if ok:
      res.append( (ret,fn,arg_types) )
  return res

def tokenize(x):
  alphanum = "abcdefghijklmnopqrstuvwxyz"
  alphanum += alphanum.upper()
  alphanum += "_.0123456789"
  toks = []
  x += "$"
  p = 0
  slashed = False
  quoted = False
  for i in range(1,len(x)):
    L,R = x[i-1], x[i]
    if L == '"' and not slashed:
      quoted = not quoted
    slashed = quoted and L == '\\'
    if quoted: continue
    if not (L in alphanum and R in alphanum): # token break
      toks.append(x[p:i])
      p = i
  return toks 

def code (code_src, **kargs):
  suff = "_nito"
  tmpdir = tempfile.mkdtemp(suffix="_" + suff)
  open(tmpdir + "/nito_inline.c","wb").write(code_src.encode("utf8"))
  libpath = tmpdir + "/nito_inline.c"
  kargs["__nito_inline__tmpdir"] = tmpdir
  return file(libpath, **kargs) 

compiler_cands = [
  ("clang", "clang --version"),
  ("gcc", "gcc --version"),
  ("gcc-8", "gcc-8 --version")
]

shim_src = """
#include <stdio.h>
#
"""

def file (libpath, **args):
  # look for the compiler
  base_cmd = None
  for opt,test in compiler_cands:
    check = "%s 1>/dev/null 2>/dev/null" % test
    if os.system(check) == 0:
      base_cmd = opt
      break
  assert base_cmd != None, "Couldn't find any match in compiler list" 

  # determine path and name for the library
  tmpdir = args.get("__nito_inline_tmpdir",None)
  suff = "_nito_inline"
  if tmpdir == None:
    tmpdir = tempfile.mkdtemp(suffix="_" + suff)
  libname = os.path.split(libpath)[-1]

  # assemble compilation flags from args or default
  defs = []
  for k,v in args.items():
    defs.append("-D%s=%s" % (k,v))
  defs = " ".join(defs)

  flags = ["std=c99","O2","march=native","ffast-math"]

  # compile to object code
  cmd = base_cmd + " "
  cmd += " ".join(["-" + x for x in flags]) + " "
  if sys.platform != "darwin": cmd += " -lm "
  cmd += defs + " -c -o %s -fpic %s"
  if sys.platform != "darwin": cmd += " -lpthread "

  # temporary files for compilation
  src_fn = libpath
  obj_fn = tmpdir + "/tmplib." + libname + ".o"
  lib_fn = tmpdir + "/tmplib." + libname + ".so"
 
  # command to link to shared library
  cmd = cmd % (obj_fn,src_fn)
  cmd += " && " + base_cmd + " -shared -o %s %s" % (lib_fn, obj_fn)

  # execute compilation/linking, exception if compiler returns nonzero
  assert os.system(cmd) == 0, "Compilation failure"

  # build the shim layer
  shim_fn = tmpdir + "/shim.c"
  open(shim_fn,"wb").write(shim_src.encode("utf8"))

  lib = Clib()
  lib.lib = ctypes.CDLL(lib_fn)
 
  # === ok, code is correct and compiled, load up the interface

  # find all function signatures
  funs = find_funcs(libpath)

  for ret, fn, arg_types in funs:
    setattr(lib,fn,getattr(lib.lib,fn))
    fp = getattr(lib,fn)
    if ret != "void": 
      fp.restype = amap[ret]
    else:
      fp.restype = None
    fp.argtypes = [amap[x] for x in arg_types]
  return lib

  # preprocess to get the macro definitions 
  cmd = base_cmd + " -E -dM -nostdinc "
  cmd += defs  
  src_fn = libpath
  out_fn = tmpdir + "/macros_" + libname + ".txt"
  cmd += "-o %s %s 2>/dev/null" % (out_fn, src_fn)
  assert os.system(cmd) == 0, "Error getting macro definitions"
  macro_defs = open(out_fn).read().split("\n")
  macros = {}
  for x in macro_defs:
    if x.strip() == "": continue
    x = x.strip()
    p = x.find(")")
    has_args = False
    if p > -1 and p < len(x)-1:
      has_args = True
    key = x.replace( "("," " ).split()[1] #.split()[1]
    p = x.find(key) + len(key)
    val = x[p:].strip()
    p = val.find(")")
    vargs = []
    if p > 0 and p+1 < len(val):
      vargs = val[:p].strip().strip("(").strip(")").split(",")
      val = val[p+1:].strip()
    macros[key] = (val, vargs, tokenize(val))

  lib.macros = macros

  return lib

