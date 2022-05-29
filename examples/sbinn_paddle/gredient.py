# code was heavily based on https://github.com/lululxvi/deepxde
# Users should be careful about adopting these functions in any commercial matters.
# https://github.com/lululxvi/deepxde#license

__all__ = ["clear", "jacobian"]

import paddle
class Jacobian:
    def __init__(self, ys, xs):
        self.ys = ys
        self.xs = xs
        self.dim_y = ys.shape[1]
        self.dim_x = xs.shape[1]
        self.J = {}

    def __call__(self, i=0, j=None):
        """Returns J[`i`][`j`]. If `j` is ``None``, returns the gradient of y_i, i.e.,
        J[i].
        """
        if not 0 <= i < self.dim_y:
            raise ValueError("i={} is not valid.".format(i))
        if j is not None and not 0 <= j < self.dim_x:
            raise ValueError("j={} is not valid.".format(j))
        # Compute J[i]
        if i not in self.J:
            y = self.ys[:, i : i + 1] if self.dim_y > 1 else self.ys
            self.J[i] = paddle.grad(
                y, self.xs,grad_outputs=paddle.ones_like(y), create_graph=False,retain_graph=True
            )[0]

        return (
                self.J[i] if j is None or self.dim_x == 1 else self.J[i][:, j : j + 1]
            )

class Jacobians:
    """Compute multiple Jacobians.

    A new instance will be created for a new pair of (output, input). For the (output,
    input) pair that has been computed before, it will reuse the previous instance,
    rather than creating a new one.
    """
    def __init__(self):
        self.Js = {}

    def __call__(self, ys, xs, i=0, j=None):

        key = (ys, xs)
        if key not in self.Js:
            self.Js[key] = Jacobian(ys, xs)
        return self.Js[key](i, j)

    def clear(self):
        """Clear cached Jacobians."""
        self.Js = {}


def jacobian(ys, xs, i=0, j=None):
    return jacobian._Jacobians(ys, xs, i=i, j=j)


jacobian._Jacobians = Jacobians()

def clear():
    """Clear cached Jacobians and Hessians."""
    jacobian._Jacobians.clear()
