import os
import sys
# Python 2/3 compatibility
if sys.version_info.major==3:
    xrange=range
    from builtins import input
    import pickle as cPickle
else:
    input=raw_input
    import cPickle
sys.path.insert(0,'./util')
from py2py3 import *
import xml_parser
import numpy as np

import data_loader

if len(sys.argv)!=2:
    print('Usage: python test_data_loader.py <config>')
    exit(0)

hyper_params=xml_parser.parse(sys.argv[1],flat=False)
data_loader_params=hyper_params['data_loader_params']
my_data_loader=data_loader.data_loader(data_loader_params)

src_folder_list2built=hyper_params['src_folder_list2built']
dest_folder_list2built=hyper_params['dest_folder_list2built']
src_folder_list2parse=hyper_params['src_folder_list2parse']
dest_folder_list2parse=hyper_params['dest_folder_list2parse']

my_data_loader.build_lists(src_folder_list2built, dest_folder_list2built, list_saved_format='pkl')
my_data_loader.build_idx_files(src_folder_list2parse, dest_folder_list2parse)

while True:

    print('type a file to parse, sh/bash to launch shell or exit to quit')
    answer=input('>>> ')

    if answer.lower() in ['exit']:
        break
    elif answer.lower() in ['sh','bash']:
        os.system('bash')
    else:
        if not os.path.exists(answer):
            print('File %s not exists!')
        else:
            origin_document, origin_summary=my_data_loader.get_raw_text(idx_file=answer)
            print('==========origin document==========')
            print(origin_document)
            print('==========origin summary===========')
            print(origin_summary)
            print('===================================')

