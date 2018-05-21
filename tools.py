import sys, os
from time import time


def create_dir(folder):
    '''
    creates a folder, if necessary
    '''
    if not os.path.exists(folder):
        os.makedirs(folder)


class TimePrint(object):
    t_last = None

    def __init__(self, text):
        TimePrint.p(text)

    @classmethod
    def p(cls, text):
        t = time()
        print text,
        if cls.t_last!=None:
            print " (took ", t-cls.t_last, "s)"
        cls.t_last = t
        sys.stdout.flush()


if __name__=="__main__":
    print "this is just a library."
