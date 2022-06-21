import paddle
import numpy as np


def minimize_bfgs(system, forces, fmax=0.5, steps=1000):
    from scipy.optimize import minimize

    if steps == 0:
        return

    if system.pos.shape[0] != 1:
        raise RuntimeError(
            "System minimization currently doesn't support replicas. Talk with Stefan to implement it."
        )

    def evalfunc(coords, info):
        coords = coords.reshape(1, -1, 3)
        coords = paddle.to_tensor(coords).astype(system.pos.dtype)
        Epot = forces.compute(coords, system.box, system.forces)[0]
        grad = -system.forces.detach().numpy().astype(np.float64)[0]
        # display information
        if info["Nfeval"] % 1 == 0:
            print(
                "{0:4d}   {1: 3.6f}   {2: 3.6f}".format(
                    info["Nfeval"], Epot, np.max(np.linalg.norm(grad, axis=1))
                )
            )
        info["Nfeval"] += 1
        return Epot, grad.reshape(-1)

    print("{0:4s} {1:9s}       {2:9s}".format("Iter", " Epot", " fmax"))
    x0 = system.pos.detach().numpy()[0].astype(np.float64)

    res = minimize(
        evalfunc,
        x0,
        method="L-BFGS-B",
        jac=True,
        options={"gtol": fmax, "maxiter": steps, "disp": False},
        args=({"Nfeval": 0},),
    )

    system.pos = paddle.to_tensor(
        res.x.reshape(1, -1, 3),
        dtype=system.pos.dtype,
        #         requires_grad=system.pos.requires_grad,
        stop_gradient=system.pos.stop_gradient,
    )


# 还未调通
def minimize_paddle_bfgs(system, forces, steps=1000):
    if steps == 0:
        return

    #     pos = system.pos.detach().requires_grad_(True)
    pos = system.pos
    pos.stop_gradient = False
    opt = paddle.optim.LBFGS([pos], max_iter=steps, tolerance_change=1e-09)

    def closure(step):
        opt.zero_grad()
        Epot = forces.compute(
            pos, system.box, system.forces, explicit_forces=False, returnDetails=False
        )
        Etot = paddle.sum(paddle.concat(Epot))
        grad = -system.forces.detach().numpy().astype(np.float64)[0]
        maxforce = float(paddle.max(paddle.norm(grad, axis=1)))
        print("{0:4d}   {1: 3.6f}   {2: 3.6f}".format(step[0], float(Etot), maxforce))
        step[0] += 1
        return Etot

    print("{0:4s} {1:9s}       {2:9s}".format("Iter", " Epot", " fmax"))
    step = [0]
    opt.step(lambda: closure(step))

    #     system.pos[:] = pos.detach().requires_grad_(False)
    system.pos = pos
    system.pos.stop_gradient = True
