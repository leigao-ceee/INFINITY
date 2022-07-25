#!/usr/bin/env python
# coding: utf-8

# # PaddleMD API tutorial

# ## System setup

# We use the `moleculekit` library for reading the input topologies and starting coordinates

# In[1]:


from moleculekit.molecule import Molecule
import os

testdir = "./test-data/prod_alanine_dipeptide_amber/"
mol = Molecule(os.path.join(testdir, "structure.prmtop"))  # Reading the system topology
mol.read(os.path.join(testdir, "input.coor"))  # Reading the initial simulation coordinates
mol.read(os.path.join(testdir, "input.xsc"))  # Reading the box dimensions

# Next we will load a forcefield file and use the above topology to extract the relevant parameters which will be used for the simulation

# In[2]:


from paddlemd.forcefields.forcefield import ForceField
from paddlemd.parameters import Parameters
import paddle

precision = paddle.float32
# device = "cuda:0"

ff = ForceField.create(mol, os.path.join(testdir, "structure.prmtop"))
parameters = Parameters(ff, mol, precision=precision)

# Now we can create a `System` object which will contain the state of the system during the simulation, including:
# 1. The current atom coordinates
# 1. The current box size
# 1. The current atom velocities
# 1. The current atom forces

# In[3]:


from paddlemd.integrator import maxwell_boltzmann
from paddlemd.systems import System

system = System(mol.numAtoms, nreplicas=1, precision=precision)
system.set_positions(mol.coords)
system.set_box(mol.box)
system.set_velocities(maxwell_boltzmann(parameters.masses, T=300, replicas=1))

# Lastly we will create a `Force` object which will be used to evaluate the potential on a given `System` state

# In[4]:


from paddlemd.forces import Forces
bonded = ["bonds", "angles", "dihedrals", "impropers", "1-4"]
# bonded = ["dihedrals"]
# forces = Forces(parameters, cutoff=9, rfa=True, switch_dist=7.5)
forces = Forces(parameters, cutoff=9, rfa=True, switch_dist=7.5, terms=bonded)
# Evaluate current energy and forces. Forces are modified in-place
Epot = forces.compute(system.pos, system.box, system.forces, returnDetails=True)

print(Epot)
print(system.forces)

# ## Dynamics

# For performing the dynamics we will create an `Integrator` object for integrating the time steps of the simulation as well as a `Wrapper` object for wrapping the system coordinates within the periodic cell

# In[5]:


from paddlemd.integrator import Integrator
from paddlemd.wrapper import Wrapper

langevin_temperature = 300  # K
langevin_gamma = 0.1
timestep = 1  # fs

integrator = Integrator(system, forces, timestep, gamma=langevin_gamma, T=langevin_temperature)
wrapper = Wrapper(mol.numAtoms, mol.bonds if len(mol.bonds) else None)

# In[6]:


from paddlemd.minimizers import minimize_bfgs

minimize_bfgs(system, forces, steps=500)  # Minimize the system steps=500

# Create a CSV file logger for the simulation which keeps track of the energies and temperature.

# In[7]:


from paddlemd.utils import LogWriter

logger = LogWriter(path="logs/", keys=('iter','ns','epot','ekin','etot','T'), name='monitor.csv')

# Now we can finally perform the full dynamics

# In[8]:


from tqdm import tqdm 
import numpy as np

FS2NS = 1E-6 # Femtosecond to nanosecond conversion

steps = 1000 # 1000
output_period = 10
save_period = 100
traj = []

trajectoryout = "mytrajectory.npy"

iterator = tqdm(range(1, int(steps / output_period) + 1))
# print(f"iterator={iterator}")
Epot = forces.compute(system.pos, system.box, system.forces)
for i in iterator:
    Ekin, Epot, T = integrator.step(niter=output_period)
    wrapper.wrap(system.pos, system.box)
#     currpos = system.pos.detach().cpu().numpy().copy()
#     currpos = system.pos.detach()
    currpos = system.pos
#     print(currpos.shape)
    traj.append(currpos)
#     print(len(traj) )
#     print(f"iterator={iterator}")
    
    if (i*output_period) % save_period  == 0:
        np.save(trajectoryout, np.stack(traj, axis=2))

    logger.write_row({'iter':i*output_period,'ns':FS2NS*i*output_period*timestep,'epot':Epot,'ekin':Ekin,'etot':Epot+Ekin,'T':T})

# In[9]:


paddle.to_tensor([2,3])

# In[10]:


system.pos.shape, system.box.shape

# In[11]:


wrapidx = paddle.to_tensor([0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10, 11, 12, 13, 14, 15, 16, 17,
        18, 19, 20, 21])

# In[12]:


pos = system.pos
# import torch
# torchpos = torch.Tensor(pos.numpy())
# torchidx = torch.Tensor([0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10, 11, 12, 13, 14, 15, 16, \
#                          18, 19, 20, 21]).int32()
# torchtmp = torchpos[:, torchidx]
# tmp = pos[:, wrapidx]
tmp = paddle.gather(pos, wrapidx, axis=1)
print(tmp.shape)
tmp1 = paddle.sum(tmp, axis=1)
print(tmp1)
# com = paddle.sum(pos[:, wrapidx], axis=1) / len(wrapidx)
