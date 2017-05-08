from __future__ import absolute_import


def toy_example(x, w):
    return [x + w]

def main(*params):
#    print 'Anything printed here will end up in the output directory for job #:', str(2)

    return toy_example(*params)
