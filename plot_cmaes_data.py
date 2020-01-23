import sys
import cma

import matplotlib
# PyCharm Professional sets other backend, which may cause trouble
if sys.platform.startswith('darwin'): # MacOS
    matplotlib.use('MacOSX')
else:
    # Not tested, may need additional dependencies.
    # See https://matplotlib.org/tutorials/introductory/usage.html#backends
    matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

if __name__ == '__main__':
    logger = cma.logger.CMADataLogger(name_prefix=sys.argv[1])
    logger.plot(fig=sys.argv[2])
    plt.show(block=True)