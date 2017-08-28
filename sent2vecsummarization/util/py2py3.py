import sys

_open=open
_map=map

def open(file,mode,encoding='latin1'):
    if mode=='r':
        if sys.version_info.major==3:
            return _open(file,mode,encoding=encoding,errors='ignore')
#        if sys.version_info.major==2:
#            import codecs
#            return codecs.open(file,mode,encoding,errors='replace')
    elif mode=='w':
        if sys.version_info.major==2:
            import codecs
            return codecs.open(file,mode,encoding,errors='ignore')
    return _open(file,mode)

def map(func,items):
    if sys.version_info.major==3:
        return list(_map(func,items))
    return _map(func,items)
