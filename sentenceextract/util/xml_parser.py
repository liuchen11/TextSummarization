import os
import sys
import xml.etree.ElementTree as et
# Python 2/3 compatibility
if sys.version_info.major==3:
    xrange=range
_open=open
_map=map
def open(file,mode):
    if mode=='r':
        if sys.version_info.major==3:
            return _open(file,mode,encoding='utf8')
        if sys.version_info.major==2:
            import codecs
            return codecs.open(file,mode,'utf8')
    return _open(file,mode)
def map(func,items):
    if sys.version_info.major==3:
        return list(_map(func,items))
    return _map(func,items)

def cast(value,tag):
    try:
        if tag=='str' or tag=='string':
            return value
        if tag in['int','enum']:
            return int(value) if not value in ['',None] else 0
        if tag=='float':
            return float(value) if not value in ['',None] else 0.0
        if tag=='bool' or tag=='boolean':
            return True if value in ['true','True'] else False
        if tag=='list_str':
            return value.split(',') if not value in ['',None] else []
        if tag=='list_int':
            return map(lambda x: cast(x,'int'),value.split(',')) if not value in ['',None] else []
        if tag=='list_float':
            return map(lambda x: cast(x,'float'),value.split(',')) if not value in ['',None] else []
        if tag=='list_bool':
            return map(lambda x: True if x in ['true','True'] else False,
                    value.split(',')) if not value in ['',None] else []
        print('unrecognized type: %s'%tag)
        return value
    except:
        raise ValueError('can not cast "%s" into type "%s"'%(value,tag))

def element2dict(element):
    ret={}
    for child in element:
        if child.getchildren()==[]:
            attr=child.attrib['name']
            category=child.attrib['type']
            value=cast(child.text,category)
            ret[attr]=value
        else:
            attr=child.tag
            ret[attr]=element2dict(child)
    return ret

def flatten(dictTree):
    ret={}
    for key in dictTree:
        if type(dictTree[key])==dict:
            subdict=flatten(dictTree[key])
            for subkey in subdict:
                if subkey in ret:
                    print('Conflict of the key value %s when flattening.'%subkey)
                else:
                    ret[subkey]=subdict[subkey]
        else:
            if key in ret:
                print('Conflict of the key value %s when flattening.'%key)
            ret[key]=dictTree[key]
    return ret

'''
>>> parse a xml file and return a dict
>>> file: str. xml file.
>>> flat: bool. use flat or hierarchical dictionary
'''
def parse(file,flat):
    root=et.parse(file).getroot()
    dictTree=element2dict(root)
    if flat:
        dictTree=flatten(dictTree)
    return dictTree

'''
>>> to print the dictionary
'''    
def print_dict(to_print,tab_num=0):
    for key in to_print:
        if type(to_print[key])==dict:
            print('\t'*tab_num,key,' =>')
            print_dict(to_print[key],tab_num+1)
        else:
            print('\t'*tab_num,key,' -> ',to_print[key])

if __name__=='__main__':
    
    if len(sys.argv)!=2:
        print('Usage: python xmlParser.py <folder>')
        exit(0)
    
    for subdir,dirs,files in os.walk(sys.argv[1]):
        results=''
        for file in files:
            if file.split('.')[-1] in ['xml','XML']:
                info=parse(subdir+os.sep+file,flat=True)
                windowLeft=info['windowLeft']
                windowWidth=info['windowWidth']
                windowRed=windowLeft+0.29*windowWidth
                results+='%f\n'%windowRed
        with open(subdir+os.sep+'results.csv','w') as fopen:
            fopen.write(results)
