import sys, os
from time import time


def create_dir(folder):
    '''
    creates a folder, if necessary
    '''
    if not os.path.exists(folder):
        os.makedirs(folder)

