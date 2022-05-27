# code was heavily based on https://github.com/lu-group/sbinn
# Users should be careful about adopting these functions in any commercial matters.
# https://github.com/lu-group/sbinn#license

import numpy as np
import paddle
import gredient as grad
import paddle.nn.functional as F
import pde
from boundary_conditions import PointSetBC
from paddlegeometry import TimeDomain
import warnings
from paddlemodel.model import Model
from paddlefnn import FNN
from paddlemodel import callbacks

paddle.seed(102)

paddle.set_default_dtype('float32')

warnings.filterwarnings('ignore')

def Variable():
    out = paddle.create_parameter(shape=(1,), dtype='float32', is_bias=True)
    return out

def sbinn(data_t, data_y, meal_t, meal_q):
    def get_variable(v, var):
        low, up = v * 0.2, v * 1.8
        l = (up - low) / 2
        v1 = l * paddle.tanh(var) + l + low
        return v1

    E_ = Variable()
    tp_ = Variable()
    ti_ = Variable()
    td_ = Variable()
    k_ = Variable()
    Rm_ = Variable()
    a1_ = Variable()
    C1_ = Variable()
    C2_ = Variable()
    C4_ = Variable()
    C5_ = Variable()
    Ub_ = Variable()
    U0_ = Variable()
    Um_ = Variable()
    Rg_ = Variable()
    alpha_ = Variable()
    beta_ = Variable()

    var_list_ = [
        E_,
        tp_,
        ti_,
        td_,
        k_,
        Rm_,
        a1_,
        C1_,
        C2_,
        C4_,
        C5_,
        Ub_,
        U0_,
        Um_,
        Rg_,
        alpha_,
        beta_,
    ]

    def ODE(t, y):
        Ip = y[:, 0:1]
        Ii = y[:, 1:2]
        G = y[:, 2:3]
        h1 = y[:, 3:4]
        h2 = y[:, 4:5]
        h3 = y[:, 5:6]

        Vp = 3
        Vi = 11
        Vg = 10
        E = (paddle.tanh(E_) + 1) * 0.1 + 0.1
        tp = (paddle.tanh(tp_) + 1) * 2 + 4
        ti = (paddle.tanh(ti_) + 1) * 40 + 60
        td = (paddle.tanh(td_) + 1) * 25 / 6 + 25 / 3
        k = get_variable(0.0083, k_)
        Rm = get_variable(209 , Rm_) 
        a1 = get_variable(6.6, a1_)
        C1 = get_variable(300, C1_)
        C2 = get_variable(144, C2_)
        C3 = 100 
        C4 = get_variable(80, C4_)
        C5 = get_variable(26 , C5_)
        Ub = get_variable(72 , Ub_)
        U0 = get_variable(4 , U0_)
        Um = get_variable(90 , Um_)
        Rg = get_variable(180 , Rg_)
        alpha = get_variable(7.5, alpha_)
        beta = get_variable(1.772, beta_)

        f1 = Rm * F.sigmoid(G / (Vg * C1) - a1)
        f2 = Ub * (1 - paddle.exp(-G / (Vg * C2)))
        kappa = (1 / Vi + 1 / (E * ti)) / C4
        f3 = (U0 + Um / (1 + paddle.pow(paddle.maximum(kappa * Ii, paddle.to_tensor(1e-3,dtype='float32')), -beta))) / (Vg * C3)
        f4 = Rg * F.sigmoid(alpha * (1 - h3 / (Vp * C5)))
        dt = t -paddle.to_tensor(meal_t,dtype='float32')
        IG = paddle.sum(
            0.5 * paddle.to_tensor(meal_q,dtype='float32') * k * paddle.exp(-k * dt) * (paddle.sign(dt) + 1),
            axis=1,
            keepdim=True,
        )
        tmp = E * (Ip / Vp - Ii / Vi)
        dIP_dt = grad.jacobian(y, t, i=0, j=0)
        dIi_dt = grad.jacobian(y, t, i=1, j=0)
        dG_dt = grad.jacobian(y, t, i=2, j=0)
        dh1_dt = grad.jacobian(y, t, i=3, j=0)
        dh2_dt = grad.jacobian(y, t, i=4, j=0)
        dh3_dt = grad.jacobian(y, t, i=5, j=0)
       
        return [
            dIP_dt - (f1 - tmp - Ip / tp),
            dIi_dt - (tmp - Ii / ti),
            dG_dt - (f4 + IG - f2 - f3 * G),
            dh1_dt - (Ip - h1) / td,
            dh2_dt - (h1 - h2) / td,
            dh3_dt - (h2 - h3) / td,
        ]
    
    geom = TimeDomain(data_t[0, 0], data_t[-1, 0])

    # Observes
    n = len(data_t)
    idx = np.append(
        np.random.choice(np.arange(1, n - 1), size=n // 5, replace=False), [0, n - 1]
    )
    observe_y2 = PointSetBC(data_t[idx], data_y[idx, 2:3], component=2)

    data = pde.PDE(geom, ODE, [observe_y2], anchors=data_t)

    net = FNN([6] + [128] * 3 + [6])

    def feature_transform(t):
        t = 0.01 * t
        return paddle.concat(
            (
                t,
                paddle.sin(t),
                paddle.sin(2 * t),
                paddle.sin(3 * t),
                paddle.sin(4 * t),
                paddle.sin(5 * t),
            ),
            axis=1,
        )

    net.apply_feature_transform(feature_transform)

    def output_transform(t, y):
        idx = 1799
        k = (data_y[idx] - data_y[0]) / (data_t[idx] - data_t[0])
        b = (data_t[idx] * data_y[0] - data_t[0] * data_y[idx]) / (
            data_t[idx] - data_t[0]
        )
        linear = paddle.to_tensor(k,dtype='float32',stop_gradient=False) * t + paddle.to_tensor(b,dtype='float32',stop_gradient=False)
        factor = paddle.tanh(t) * paddle.tanh(idx - t)
      
        return  linear + factor * paddle.to_tensor([1, 1, 1e2, 1, 1, 1],dtype='float32') * y
        
    net.apply_output_transform(output_transform)

    model = Model(data, net)

    firsttrain = 10000
    callbackperiod = 1000
    maxepochs = 600000

    model.compile("adam", lr=1e-3, loss_weights=[0, 0, 0, 0, 0, 0, 0.01])
    model.train(epochs=firsttrain, display_every=1000)

    model.compile(
        "adam",
        lr=1e-7,
        loss_weights=[1, 1, 1e-2, 1, 1, 1, 1e-2],
        external_trainable_variables=var_list_,
    )

    variablefilename = "variables.csv"
    variable = callbacks.VariableValue(
        var_list_, period=callbackperiod, filename=variablefilename
    )
    losshistory, train_state = model.train(
        epochs=maxepochs, display_every=1000, callbacks=[variable]
    )

gluc_data = np.hsplit(np.loadtxt("glucose_gen.dat"), [1])
meal_data = np.hsplit(np.loadtxt("meal_gen.dat"), [4])

t = gluc_data[0]
y = gluc_data[1]
meal_t = meal_data[0]
meal_q = meal_data[1]

sbinn(t[:1800], y[:1800], meal_t, meal_q)


