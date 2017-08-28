import os
import sys
if sys.version_info.major==2:
    import cPickle
else:
    import pickle as cPickle

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def plot_loss_curve(pkl_file):
    loss_list=cPickle.load(open(pkl_file, 'rb'))
    cleaned_loss_list=[]
    for ckpt, loss in loss_list:
        ckpt_pt=int(ckpt.split('-')[-1])
        cleaned_loss_list.append([ckpt_pt, loss])

    if sys.version_info.major==2:
        cleaned_loss_list=sorted(cleaned_loss_list, lambda x,y: -1 if x[0]<y[0] else 1)
    else:
        cleaned_loss_list=sorted(cleaned_loss_list, key=lambda x:x[0], reverse=False)

    ckpt_pt_list=map(lambda x:x[0], cleaned_loss_list)
    ckpt_loss_list=map(lambda x:x[1], cleaned_loss_list)

    plt.plot(ckpt_pt_list, ckpt_loss_list)
    plt.xlabel('Batches')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.show()


if __name__=='__main__':

    if len(sys.argv)<2:
        print('Usage: python plot_loss_curve.py <pkl_file>')
        exit(0)

    pkl_file=sys.argv[1]
    plot_loss_curve(pkl_file=pkl_file)


