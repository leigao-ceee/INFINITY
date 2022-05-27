# code was heavily based on https://github.com/lululxvi/deepxde
# Users should be careful about adopting these functions in any commercial matters.
# https://github.com/lululxvi/deepxde#license

__all__ = ["get", "is_external_optimizer"]

from ..config import LBFGS_options
import paddle

def is_external_optimizer(optimizer):
    return optimizer in ["L-BFGS", "L-BFGS-B"]


def get(params, optimizer, learning_rate=None, decay=None):
    """Retrieves an Optimizer instance."""
    if isinstance(optimizer, paddle.optimizer.Optimizer):
        return optimizer

    if optimizer in ["L-BFGS", "L-BFGS-B"]:
        if learning_rate is not None or decay is not None:
            print("Warning: learning rate is ignored for {}".format(optimizer))
        return None

    if learning_rate is None:
        raise ValueError("No learning rate for {}.".format(optimizer))

    if decay is not None:
        # TODO: learning rate decay
        raise NotImplementedError(
            "learning rate decay to be implemented for backend pytorch."
        )
    if optimizer == "adam":
        return paddle.optimizer.Adam(parameters=params,learning_rate=learning_rate)
    raise NotImplementedError(f"{optimizer} to be implemented for backend paddle.")