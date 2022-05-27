# code was heavily based on https://github.com/lululxvi/deepxde
# Users should be careful about adopting these functions in any commercial matters.
# https://github.com/lululxvi/deepxde#license

import numpy as np

from .geometry_1d import Interval

class TimeDomain(Interval):
    def __init__(self, t0, t1):
        super().__init__(t0, t1)
        self.t0 = t0
        self.t1 = t1

    def on_initial(self, t):
        return np.isclose(t, self.t0).flatten()

if __name__ == '__main__':
    tt = TimeDomain(0,1799)
    print(tt.on_initial(np.array([1,2,3,4,5])))