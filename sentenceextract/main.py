import os
import sys
import socket
import select
import traceback
import nltk.data

sys.path.insert(0,'./run')
sys.path.insert(0,'./util')

from py2py3 import *

import html
import laucher
import xml_parser

def encode_sth(item):
    coding=['iso-8859-1','utf8','latin1','ascii']
    for coding_format in coding:
        try:
            coded=item.encode(coding_format)
            return coded
        except:
            continue
    raise Exception('Unable to encode',item)

def decode_sth(item):
    coding=['iso-8859-1','utf8','latin1','ascii']
    for coding_format in coding:
        try:
            coded=item.decode(coding_format)
            return coded
        except:
            continue
    raise Exception('Unable to decode',item)


if __name__=='__main__':

    if len(sys.argv)!=2:
        print('Usage: python main.py <config>')
        exit(0)

    # laucher construction
    laucher_params=xml_parser.parse(sys.argv[1],flat=False)
    my_laucher=laucher.laucher(laucher_params)
    my_laucher.start()

    # set up the server
    host='127.0.0.1'
    port=8100

    socket_list=[]
    try:
        server=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        server.bind((host,port))
        server.listen(5)
        socket_list.append(server)
        print('Server/Client mode start! Listening on %s:%s'%(host,port))
    except:
        traceback.print_exc()
        print('Unable to start the server. Abort!')
        exit(1)

    while True:
        ready2read,ready2write,in_err=select.select(socket_list,[],[],0)

        for sock in ready2read:
            if sock==server:        # New connection
                sockfd,addr=server.accept()
                socket_list.append(sockfd)
                print('Client (%s,%s) connected'%addr)
            else:                   # Message from a client
                try:
                    data=encode_sth(sock.recv(1024))
                    if data:
                        try:
                            data=data.rstrip('\n')
                            print('Analyzing content on %s'%data)
                            raw_content=html.get_content_from_url(data)
                            raw_sentences=raw_content.split('\n')
                            text_content_list=[]
                            for sentence in raw_sentences:
                                if len(sentence.split(' '))>=5:
                                    text_content_list.append(sentence)
                            text_content='\n'.join(text_content_list)
                            text_content_list_with_idx=[]
                            for idx,sentence in enumerate(text_content.split('\n')):
                                text_content_list_with_idx.append('[%d] %s'%(idx+1,sentence))
                            content_with_idx='\n'.join(text_content_list_with_idx)
                            if len(text_content_list)==0:
                                message='%s@@@@@%s'%(encode_sth('No content extracted.'),encode_sth('No summary generated.'))
                                sock.send(message)
                                print('Completed!')
                                continue
                            elif len(text_content_list)<5:
                                message='%s@@@@@%s'%(encode_sth(content_with_idx),encode_sth('Text too short to summarize.'))
                                sock.send(message)
                                print('Completed!')
                                continue
                            with open('tmp.txt','w') as fopen:
                                fopen.write(text_content)
                            output=my_laucher.run('tmp.txt')
                            os.system('rm tmp.txt')
                            message='%s@@@@@%s'%(encode_sth(content_with_idx),output)
                            sock.send(message)
                        except:
                            traceback.print_exc()
                            message='Oops, some problems occurs while loading contents from %s'%data
                            print(message)
                            sock.send(encode_sth(message+'\n'))
                            continue
                    else:
                        if sock in socket_list:
                            socket_list.remove(sock)
                        print('One server is offline')
                except:
                    traceback.print_exc()
                    print('Some error happens')
                    break

    server.close()
