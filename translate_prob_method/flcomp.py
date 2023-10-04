import sys
import numpy as np
import scipy.stats
import os

def load_scores(input_file):
    with open(input_file) as f:
        return np.array([float(n) for n in f.read().split("\n") if n != ""])


def main(argv):
    _, input_file = argv
    zip_name = input_file.split('/')[-1]+'.zip'
    type_name = ""
    f=open(f'description.txt','w+')
    f.close()
    os.system(f'cp {input_file} seg.scores')
    os.system(f'zip -r {zip_name}  seg.scores description.txt')
    os.system(f'rm seg.scores')
    os.system(f'rm description.txt')



# Run
if __name__ == '__main__':
    main(sys.argv)
