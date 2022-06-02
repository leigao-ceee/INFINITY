import sys
sys.path.append('./')
import lettuce as lt
import paddle
import numpy as np
import matplotlib.pyplot as plt


paddle.set_device('gpu')
device = ''  # replace with device("cpu"), if no GPU is available

def EnergyReporter(lattice, flow, interval=1, starting_iteration=0, out=sys.stdout):
    from lettuce.observables import IncompressibleKineticEnergy
    return lt.ObservableReporter(IncompressibleKineticEnergy(lattice, flow), interval=interval, out=out)
print("start")

def run(ny=100, *axes):

    lattice = lt.Lattice(lt.D2Q9, device=device, dtype=paddle.float32)  # single precision - float64 for double precision

    ny = 128
    # resolution = 2 * ny  # resolution of the lattice, low resolution leads to unstable speeds somewhen after 10 (PU)

    flow = lt.Obstacle2D(2*ny, ny,50, 0.05, lattice,10.1)

    x, y = flow.grid
    flow.mask = ((x >= 2) & (x < 3) &  (y >= x) & (y <= 3))
    axes[0].imshow(flow.mask.T, origin="lower")

    tau = flow.units.relaxation_parameter_lu 
    sim = lt.Simulation(flow, lattice, lt.BGKCollision(lattice, tau), 
                            lt.StandardStreaming(lattice))
    sim.step(ny * 100)
    u = flow.units.convert_velocity_to_pu(lattice.u(sim.f).detach().cpu().numpy())
    return axes[1].imshow(u[0,...].T, origin="lower")
    print("Max Velocity:", u.max())

def run_and_plot(n):
    fig, axes = plt.subplots(1,2, figsize=(10,3))
    fig.subplots_adjust(right=0.85)
    im2 = run(n, *axes)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
    fig.colorbar(im2, cax=cbar_ax)
    fig_name = './Resolution_' + str(n) +'.png'
    plt.savefig(fig_name)


if __name__ == '__main__':
    run_and_plot(25)
    run_and_plot(50)
    run_and_plot(100)

