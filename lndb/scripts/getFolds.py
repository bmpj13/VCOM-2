import numpy as np
import random
import csv
from utils import readCsv, writeCsv

def getFold(fold = 0, fname_in = '../trainset_csv/trainFolds.csv',
            fnames = ['CTs.csv','Fleischner.csv','Nodules.csv'],
            prefix_in = '../trainset_csv/train', prefix_out = '',
            excludeFold = False):
    
    if not prefix_out:
        prefix_out = '../folds/fold{}_'.format(fold)
    
    #Get fold lnds
    nodules = readCsv(fname_in)
    header = nodules[0]
    lines = nodules[1:]
    
    foldind = header.index('Fold{}'.format(fold))
    foldlnd = [l[foldind] for l in lines if len(l)>foldind]
    
    for fname in fnames:
        lines = readCsv(prefix_in+fname)
        header = lines[0]
        lines = lines[1:]
        
        lndind = header.index('LNDbID')
        if not excludeFold:
            lines = [l for l in lines if l[lndind] in foldlnd]
        else:
            lines = [l for l in lines if not l[lndind] in foldlnd]
        
        #Save to csv
        writeCsv(prefix_out+fname,[header]+lines)
            
if __name__ == "__main__":
    # Get fold 0 from trainset
    # getFold(fold=0)
    getFold(fold=0, fnames=['Nodules.csv'])
    getFold(fold=1, fnames=['Nodules.csv'])
    getFold(fold=2, fnames=['Nodules.csv'])
    getFold(fold=3, fnames=['Nodules.csv'])