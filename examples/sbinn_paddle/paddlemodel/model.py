# code was heavily based on https://github.com/lululxvi/deepxde
# Users should be careful about adopting these functions in any commercial matters.
# https://github.com/lululxvi/deepxde#license

__all__ = ["Model", "TrainState", "LossHistory"]

import numpy as np
import paddle
import sys
sys.path.append('paddlemodel')
sys.path.append('../')
import config
import gredient as grad
from putils import *
import display
import losses as losses_module
import metrics as metrics_module
from callbacks import CallbackList
from optimizers.paddle import optimizers


class Model:
    """A ``Model`` trains a ``NN`` on a ``Data``.

    Args:
        data: ``deepxde.data.Data`` instance.
        net: ``deepxde.nn.NN`` instance.
    """

    def __init__(self, data, net):
        self.data = data
        self.net = net

        self.opt_name = None
        self.batch_size = None
        self.callbacks = None
        self.metrics = None
        self.external_trainable_variables = []
        self.train_state = TrainState()
        self.losshistory = LossHistory()
        self.stop_training = False

        # Backend-dependent attributes
        self.opt = None
        # Tensor or callable
        self.outputs = None
        self.outputs_losses = None
        self.train_step = None
        self.pre_loss = None

    def compile(
        self,
        optimizer,
        lr=None,
        loss="MSE",
        metrics=None,
        decay=None,
        loss_weights=None,
        external_trainable_variables=None,
    ):
        """Configures the model for training.

        Args:
            optimizer: String. Name of optimizer.
            lr: A Tensor or a floating point value. The learning rate. For L-BFGS, use
                `dde.optimizers.set_LBFGS_options` to set the hyperparameters.
            loss: If the same loss is used for all errors, then `loss` is a String (name
                of objective function) or objective function. If different errors use
                different losses, then `loss` is a list whose size is equal to the
                number of errors.
            metrics: List of metrics to be evaluated by the model during training.
            decay: Tuple. Name and parameters of decay to the initial learning rate. One
                of the following options:

                - `inverse time decay <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/InverseTimeDecay>`_: ("inverse time", decay_steps, decay_rate)
                - `cosine decay <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/CosineDecay>`_: ("cosine", decay_steps, alpha)

            loss_weights: A list specifying scalar coefficients (Python floats) to
                weight the loss contributions. The loss value that will be minimized by
                the model will then be the weighted sum of all individual losses,
                weighted by the loss_weights coefficients.
            external_trainable_variables: A trainable ``tf.Variable`` object or a list
                of trainable ``tf.Variable`` objects. The unknown parameters in the
                physics systems that need to be recovered. If the backend is
                tensorflow.compat.v1, `external_trainable_variables` is ignored, and all
                trainable ``tf.Variable`` objects are automatically collected.
        """
        print("Compiling model...")

        self.opt_name = optimizer
        loss_fn = losses_module.get(loss)
        if external_trainable_variables is None:
            self.external_trainable_variables = []
        else:
            self.external_trainable_variables = external_trainable_variables

        self._compile_paddle(lr, loss_fn, decay, loss_weights)

        # metrics may use model variables such as self.net, and thus are instantiated
        # after backend compile.
        metrics = metrics or []
        self.metrics = [metrics_module.get(m) for m in metrics]
    def _compile_paddle(self, lr, loss_fn, decay, loss_weights):


        def outputs(training, inputs):
            self.net.train(mode=training)
            with paddle.no_grad():
                return self.net(paddle.to_tensor(inputs,dtype='float32'))

        def outputs_losses(training, inputs, targets):
            if training:
                self.net.train()
            else:
                self.net.eval()
            self.net.inputs = paddle.to_tensor(inputs,dtype='float32',stop_gradient=False)
            # self.net.inputs.requires_grad_()
            # self.net.inputs = paddle.to_tensor(self.net.inputs,stop_gradient=False)
            outputs_ = self.net(self.net.inputs)
            # print(outputs_)
            # Data losses
            if targets is not None:
                targets = paddle.to_tensor(targets,dtype='float32',stop_gradient=False)
            # targets = 5*self.net.inputs
            # losses = paddle.nn.MSELoss()(self.net.inputs,targets)
            losses = self.data.losses(targets, outputs_, loss_fn, self)
            if not isinstance(losses, list):
                losses = [losses]
            # TODO: regularization
            losses = paddle.stack(losses)
            losses = losses.squeeze(1)

            # Weighted losses
            if loss_weights is not None:
                losses *= paddle.to_tensor(loss_weights,dtype='float32')
                self.losshistory.set_loss_weights(loss_weights)
            # Clear cached Jacobians and Hessians.
           

            grad.clear()
            return outputs_, losses

        print(self.net)
        trainable_variables = (
            list(self.net.parameters()) + self.external_trainable_variables
        )
        self.opt = optimizers.get(
            trainable_variables, self.opt_name, learning_rate=lr, decay=decay
        )
        def train_step(inputs, targets):
            # if self.train_state.epoch %1000==0:
            #     print('weight',self.net.linears[0].weight.transpose((1,0))[0])
            def closure():
                losses = outputs_losses(True, inputs, targets)[1]
                total_loss = paddle.sum(losses)
                self.opt.clear_grad()
                total_loss.backward()
                self.opt.step()
                # print(total_loss)
                return total_loss
            # self.opt.step(closure)
            closure()
            # if self.train_state.epoch %1000==0:
            #     print('weight_grad',self.net.linears[0].weight.grad.transpose((1,0))[0])
            #     print('weight',self.net.linears[0].weight.transpose((1,0))[0])
        # Callables
        self.outputs = outputs
        self.outputs_losses = outputs_losses
        self.train_step = train_step

    def _outputs(self, training, inputs):
        outs = self.outputs(training, inputs)
        return outs.detach().cpu().numpy()

    def _outputs_losses(self, training, inputs, targets, auxiliary_vars):
        self.net.requires_grad_(requires_grad=False)
        outs = self.outputs_losses(training, inputs, targets)
        self.net.requires_grad_(requires_grad=True)
        # return outs.detach().cpu().numpy()
        return [out.detach().cpu().numpy() for out in outs]

    def _train_step(self, inputs, targets, auxiliary_vars):
        self.train_step(inputs, targets)
        

    def train(
        self,
        epochs=None,
        batch_size=None,
        display_every=1000,
        disregard_previous_best=False,
        callbacks=None,
        model_restore_path=None,
        model_save_path=None,
    ):
        """Trains the model for a fixed number of epochs (iterations on a dataset).

        Args:
            epochs: Integer. Number of iterations to train the model. Note: It is the
                number of iterations, not the number of epochs.
            batch_size: Integer or ``None``. If you solve PDEs via ``dde.data.PDE`` or
                ``dde.data.TimePDE``, do not use `batch_size`, and instead use
                `dde.callbacks.PDEResidualResampler
                <https://deepxde.readthedocs.io/en/latest/modules/deepxde.html#deepxde.callbacks.PDEResidualResampler>`_,
                see an `example <https://github.com/lululxvi/deepxde/blob/master/examples/diffusion_1d_resample.py>`_.
            display_every: Integer. Print the loss and metrics every this steps.
            disregard_previous_best: If ``True``, disregard the previous saved best
                model.
            callbacks: List of ``dde.callbacks.Callback`` instances. List of callbacks
                to apply during training.
            model_restore_path: String. Path where parameters were previously saved.
                See ``save_path`` in `tf.train.Saver.restore <https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/Saver#restore>`_.
            model_save_path: String. Prefix of filenames created for the checkpoint.
                See ``save_path`` in `tf.train.Saver.save <https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/Saver#save>`_.
        """
        self.batch_size = batch_size
        self.callbacks = CallbackList(callbacks=callbacks)
        self.callbacks.set_model(self)
        if disregard_previous_best:
            self.train_state.disregard_best()

        if model_restore_path is not None:
            self.restore(model_restore_path, verbose=1)
        print("Training model...\n")
        self.stop_training = False
        self.train_state.set_data_train(*self.data.train_next_batch(self.batch_size))
        self.train_state.set_data_test(*self.data.test())
        self._test()
        self.callbacks.on_train_begin()
        if optimizers.is_external_optimizer(self.opt_name):
            self._train_paddle_lbfgs()
        else:
            if epochs is None:
                raise ValueError("No epochs for {}.".format(self.opt_name))
            self._train_sgd(epochs, display_every)
        self.callbacks.on_train_end()

        display.training_display.summary(self.train_state)
        if model_save_path is not None:
            self.save(model_save_path, verbose=1)
        return self.losshistory, self.train_state

    def _train_sgd(self, epochs, display_every):
        for i in range(epochs):
            self.callbacks.on_epoch_begin()
            self.callbacks.on_batch_begin()

            self.train_state.set_data_train(
                *self.data.train_next_batch(self.batch_size)
            )
            self._train_step(
                self.train_state.X_train,
                self.train_state.y_train,
                self.train_state.train_aux_vars,
            )

            self.train_state.epoch += 1
            self.train_state.step += 1
            if self.train_state.step % display_every == 0 or i + 1 == epochs:
                self._test()

            self.callbacks.on_batch_end()
            self.callbacks.on_epoch_end()

            if self.stop_training:
                break

    def _train_paddle_lbfgs(self):
        prev_n_iter = 0
        while prev_n_iter < optimizers.LBFGS_options["maxiter"]:
            self.callbacks.on_epoch_begin()
            self.callbacks.on_batch_begin()

            self.train_state.set_data_train(
                *self.data.train_next_batch(self.batch_size)
            )
            self._train_step(
                self.train_state.X_train,
                self.train_state.y_train,
                self.train_state.train_aux_vars,
            )

            n_iter = self.opt.state_dict()["state"][0]["n_iter"]
            if prev_n_iter == n_iter:
                # Converged
                break

            self.train_state.epoch += n_iter - prev_n_iter
            self.train_state.step += n_iter - prev_n_iter
            prev_n_iter = n_iter
            self._test()

            self.callbacks.on_batch_end()
            self.callbacks.on_epoch_end()

            if self.stop_training:
                break

    def _test(self):
        (
            self.train_state.y_pred_train,
            self.train_state.loss_train,
        ) = self._outputs_losses(
            True,
            self.train_state.X_train,
            self.train_state.y_train,
            self.train_state.train_aux_vars,
        )
        self.train_state.y_pred_test, self.train_state.loss_test = self._outputs_losses(
            False,
            self.train_state.X_test,
            self.train_state.y_test,
            self.train_state.test_aux_vars,
        )

        if isinstance(self.train_state.y_test, (list, tuple)):
            self.train_state.metrics_test = [
                m(self.train_state.y_test[i], self.train_state.y_pred_test[i])
                for m in self.metrics
                for i in range(len(self.train_state.y_test))
            ]
        else:
            self.train_state.metrics_test = [
                m(self.train_state.y_test, self.train_state.y_pred_test)
                for m in self.metrics
            ]

        self.train_state.update_best()
        self.losshistory.append(
            self.train_state.step,
            self.train_state.loss_train,
            self.train_state.loss_test,
            self.train_state.metrics_test,
        )

        if (
            np.isnan(self.train_state.loss_train).any()
            or np.isnan(self.train_state.loss_test).any()
        ):
            self.stop_training = True

        display.training_display(self.train_state)

    def predict(self, x, operator=None, callbacks=None):
        if isinstance(x, tuple):
            x = tuple(np.array(xi, dtype=config.real(np)) for xi in x)
        else:
            x = np.array(x, dtype=config.real(np))
        self.callbacks = CallbackList(callbacks=callbacks)
        self.callbacks.set_model(self)
        self.callbacks.on_predict_begin()
        if operator is None:
            y = self._outputs(False, x)
            self.callbacks.on_predict_end()
            return y

        # operator is not None
        if get_num_args(operator) == 3:
            aux_vars = self.data.auxiliary_var_fn(x).astype(config.real(np))
        self.net.eval()
        inputs = paddle.to_tensor(x,dtype='float32')
        inputs.requires_grad_()
        outputs = self.net(inputs)
        if get_num_args(operator) == 2:
            y = operator(inputs, outputs)
        elif get_num_args(operator) == 3:
            raise NotImplementedError(
                "Model.predict() with auxiliary variable hasn't been implemented."
            )
        y = y.detach().cpu().numpy()
        self.callbacks.on_predict_end()
        return y

    def state_dict(self):
        """Returns a dictionary containing all variables."""
        # TODO: backend tensorflow
        destination = self.net.state_dict()
        return destination

    def save(self, save_path):
        save_path = f"{save_path}-{self.train_state.epoch}"
        save_path += ".pt"
        checkpoint = {
            "model_state_dict": self.net.state_dict(),
            "optimizer_state_dict": self.opt.state_dict(),
        }
        paddle.save(checkpoint, save_path)
        return save_path
    def restore(self, save_path):
        checkpoint = paddle.load(save_path)
        self.net.load_state_dict(checkpoint["model_state_dict"])
        self.opt.load_state_dict(checkpoint["optimizer_state_dict"])

class TrainState:
    def __init__(self):
        self.epoch = 0
        self.step = 0

        # Current data
        self.X_train = None
        self.y_train = None
        self.train_aux_vars = None
        self.X_test = None
        self.y_test = None
        self.test_aux_vars = None

        # Results of current step
        # Train results
        self.loss_train = None
        self.y_pred_train = None
        # Test results
        self.loss_test = None
        self.y_pred_test = None
        self.y_std_test = None
        self.metrics_test = None

        # The best results correspond to the min train loss
        self.best_step = 0
        self.best_loss_train = np.inf
        self.best_loss_test = np.inf
        self.best_y = None
        self.best_ystd = None
        self.best_metrics = None

    def set_data_train(self, X_train, y_train, train_aux_vars=None):
        self.X_train = X_train
        self.y_train = y_train
        self.train_aux_vars = train_aux_vars

    def set_data_test(self, X_test, y_test, test_aux_vars=None):
        self.X_test = X_test
        self.y_test = y_test
        self.test_aux_vars = test_aux_vars

    def update_best(self):
        if self.best_loss_train > np.sum(self.loss_train):
            self.best_step = self.step
            self.best_loss_train = np.sum(self.loss_train)
            self.best_loss_test = np.sum(self.loss_test)
            self.best_y = self.y_pred_test
            self.best_ystd = self.y_std_test
            self.best_metrics = self.metrics_test

    def disregard_best(self):
        self.best_loss_train = np.inf

class LossHistory:
    def __init__(self):
        self.steps = []
        self.loss_train = []
        self.loss_test = []
        self.metrics_test = []
        self.loss_weights = 1

    def set_loss_weights(self, loss_weights):
        self.loss_weights = loss_weights

    def append(self, step, loss_train, loss_test, metrics_test):
        self.steps.append(step)
        self.loss_train.append(loss_train)
        if loss_test is None:
            loss_test = self.loss_test[-1]
        if metrics_test is None:
            metrics_test = self.metrics_test[-1]
        self.loss_test.append(loss_test)
        self.metrics_test.append(metrics_test)