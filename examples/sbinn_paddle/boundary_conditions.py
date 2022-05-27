# code was heavily based on https://github.com/lululxvi/deepxde
# Users should be careful about adopting these functions in any commercial matters.
# https://github.com/lululxvi/deepxde#license

"""Boundary conditions."""

import paddle
import numbers
from abc import ABC, abstractmethod
from functools import wraps
import numpy as np
import config
import gredient as grad

class BC(ABC):
    """Boundary condition base class.

    Args:
        geom: A ``deepxde.geometry.Geometry`` instance.
        on_boundary: A function: (x, Geometry.on_boundary(x)) -> True/False.
        component: The output component satisfying this BC.
    """

    def __init__(self, geom, on_boundary, component):
        self.geom = geom
        self.on_boundary = lambda x, on: np.array(
            [on_boundary(x[i], on[i]) for i in range(len(x))]
        )
        self.component = component

        self.boundary_normal = npfunc_range_autocache(
            paddle.to_tensor(self.geom.boundary_normal,dtype='float32')
        )

    def filter(self, X):
        return X[self.on_boundary(X, self.geom.on_boundary(X))]

    def collocation_points(self, X):
        return self.filter(X)

    def normal_derivative(self, X, inputs, outputs, beg, end):
        dydx = grad.jacobian(outputs, inputs, i=self.component, j=None)[beg:end]
        n = self.boundary_normal(X, beg, end)
        return paddle.sum(dydx * n, 1, keepdims=True)
    @abstractmethod
    def error(self, X, inputs, outputs, beg, end):
        """Returns the loss."""

class DirichletBC(BC):
    """Dirichlet boundary conditions: y(x) = func(x)."""

    def __init__(self, geom, func, on_boundary, component=0):
        super().__init__(geom, on_boundary, component)
        self.func = npfunc_range_autocache(paddle.to_tensor(func,dtype='float32'))

    def error(self, X, inputs, outputs, beg, end):
        values = self.func(X, beg, end)
        if len(values.shape) > 0 and values.shape[1] != 1:
            raise RuntimeError(
                "DirichletBC func should return an array of shape N by 1 for a single"
                " component. Use argument 'component' for different components."
            )
        return outputs[beg:end, self.component : self.component + 1] - values


class NeumannBC(BC):
    """Neumann boundary conditions: dy/dn(x) = func(x)."""

    def __init__(self, geom, func, on_boundary, component=0):
        super().__init__(geom, on_boundary, component)
        self.func = npfunc_range_autocache(paddle.to_tensor(func,dtype='float32'))

    def error(self, X, inputs, outputs, beg, end):
        values = self.func(X, beg, end)
        return self.normal_derivative(X, inputs, outputs, beg, end) - values


class RobinBC(BC):
    """Robin boundary conditions: dy/dn(x) = func(x, y)."""

    def __init__(self, geom, func, on_boundary, component=0):
        super().__init__(geom, on_boundary, component)
        self.func = func

    def error(self, X, inputs, outputs, beg, end):
        return self.normal_derivative(X, inputs, outputs, beg, end) - self.func(
            X[beg:end], outputs[beg:end]
        )


class PeriodicBC(BC):
    """Periodic boundary conditions on component_x."""

    def __init__(self, geom, component_x, on_boundary, derivative_order=0, component=0):
        super().__init__(geom, on_boundary, component)
        self.component_x = component_x
        self.derivative_order = derivative_order
        if derivative_order > 1:
            raise NotImplementedError(
                "PeriodicBC only supports derivative_order 0 or 1."
            )

    def collocation_points(self, X):
        X1 = self.filter(X)
        X2 = self.geom.periodic_point(X1, self.component_x)
        return np.vstack((X1, X2))

    def error(self, X, inputs, outputs, beg, end):
        mid = beg + (end - beg) // 2
        if self.derivative_order == 0:
            yleft = outputs[beg:mid, self.component : self.component + 1]
            yright = outputs[mid:end, self.component : self.component + 1]
        else:
            dydx = grad.jacobian(outputs, inputs, i=self.component, j=self.component_x)
            yleft = dydx[beg:mid]
            yright = dydx[mid:end]
        return yleft - yright


class OperatorBC(BC):
    def __init__(self, geom, func, on_boundary):
        super().__init__(geom, on_boundary, 0)
        self.func = func

    def error(self, X, inputs, outputs, beg, end):
        return self.func(inputs, outputs, X)[beg:end]


class PointSetBC:
    def __init__(self, points, values, component=0):
        self.points = np.array(points, dtype=config.real(np))
        if not isinstance(values, numbers.Number) and values.shape[1] != 1:
            raise RuntimeError(
                "PointSetBC should output 1D values. Use argument 'component' for different components."
            )
        self.values = paddle.to_tensor(values, dtype='float32')
        self.component = component

    def collocation_points(self, X):
        return self.points

    def error(self, X, inputs, outputs, beg, end):
        # return outputs[beg:end, self.component : self.component + 1] - self.values
        return outputs[int(beg):int(end), int(self.component) : int(self.component) + 1] - self.values


def npfunc_range_autocache(func):
    cache = {}

    @wraps(func)
    def wrapper_nocache(X, beg, end):
        return func(X[beg:end])

    @wraps(func)
    def wrapper_cache(X, beg, end):
        key = (id(X), beg, end)
        if key not in cache:
            cache[key] = func(X[beg:end])
        return cache[key]


    return wrapper_cache

if __name__ == '__main__':
    t = np.array([[1],[2],[3],[4]])
    y = np.array([[1], [2], [3], [4]])
    ob = PointSetBC(t,y)
    print(ob.values)
    pass
