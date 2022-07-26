#!/usr/bin/env python
# coding: utf-8

# # 开始复现paddlemd
# 
# 首先moleculekit包在AIStudio下安装不成功，需要使用右侧的包管理，手动点击安装。
# 用包管理也没有安装成功，后来是下载源码，编译安装成功的。
# 
# [tutorial](tutorial.ipynb)

# In[11]:


# !cat /home/aistudio/.webide/3863645/log/install-moleculekit-2022-04-22-22-13-08.log
# !git clone https://github.com/Acellera/moleculekit

# # 第一阶段 手工开始单个文件转换
# 将项目所有核心.py文件，使用`%%writefile xx.py`的格式，放到notebook cell中，这样可以通过查找替换，快速修改所有的代码。
# * 优点是：代码修改效率高。发现一个问题，解决问题，并可以全部查找、替换，将同类问题全部解决。
# * 缺点是：调试效率较低。需要另开一个notebook文件进行测试，且修改代码后，需要重新执行，甚至要重启测试项目的内核。

# # 代码复现第二阶段
# 
# 像常规notebook下的调试流程
# ##  1、对疑点文件拆分，将函数放到Cell进行测试
# 
# 测试中可以加入测试代码，验证函数是否正确。最终保证所有函数测试通过
# ##  2、测试通过后，将修改写回文件
# ## 3、在tutorial.ipynb文件中总测试
# 
# 优点是，基本不修改tutorial.ipynb文件代码。

# # 开始第二阶段调试

# Next we will load a forcefield file and use the above topology to extract the relevant parameters which will be used for the simulation
# 

# In[12]:


# 加入计算时间代码
import time
class Timer:  #@save
    """记录多次运行时间"""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """启动计时器"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间"""
        return np.array(self.times).cumsum().tolist()


class Benchmark:
    """用于测量运行时间"""
    def __init__(self, description='Done'):
        self.description = description

    def __enter__(self):
        self.timer = Timer()
        return self

    def __exit__(self, *args):
        print(f'{self.description}: {self.timer.stop():.4f} sec')

# In[13]:


with Benchmark("测试速度"):
    print("hello world")
    tmp = 0
print(tmp)

# # paddlemd/forcefields/forcefield.py

# In[14]:


# ./paddlemd/forcefields/forcefield.py
from abc import ABC, abstractmethod
import os


class _ForceFieldBase(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_atom_types(self):
        pass

    @abstractmethod
    def get_charge(self, at):
        pass

    @abstractmethod
    def get_mass(self, at):
        pass

    @abstractmethod
    def get_LJ(self, at):
        pass

    @abstractmethod
    def get_bond(self, at1, at2):
        pass

    @abstractmethod
    def get_angle(self, at1, at2, at3):
        pass

    @abstractmethod
    def get_dihedral(self, at1, at2, at3, at4):
        pass

    @abstractmethod
    def get_14(self, at1, at2, at3, at4):
        pass

    @abstractmethod
    def get_improper(self, at1, at2, at3, at4):
        pass


class ForceField:
    def create(mol, prm):
        from paddlemd.forcefields.ff_yaml import YamlForcefield
        from paddlemd.forcefields.ff_parmed import ParmedForcefield

        parmedext = [".prm", ".prmtop", ".frcmod"]
        yamlext = [".yaml", ".yml"]
        if isinstance(prm, str):
            ext = os.path.splitext(prm)[-1]
            if ext in parmedext:
                return ParmedForcefield(mol, prm)
            elif ext in yamlext:
                return YamlForcefield(mol, prm)
            else:  # Fallback on parmed
                return ParmedForcefield(mol, prm)
        else:  # Fallback on parmed
            return ParmedForcefield(mol, prm)


# In[15]:


# ./paddlemd/forcefields/ff_yaml.py
# from paddlemd.forcefields.forcefield import _ForceFieldBase
from math import radians
import numpy as np
import yaml


class YamlForcefield(_ForceFieldBase):
    def __init__(self, mol, prm):
        self.mol = mol
        with open(prm, "r") as f:
            self.prm = yaml.load(f, Loader=yaml.FullLoader)

    def _get_x_variants(self, atomtypes):
        from itertools import product

        permutations = np.array(
            sorted(
                list(product([False, True], repeat=len(atomtypes))),
                key=lambda x: sum(x),
            )
        )
        variants = []
        for per in permutations:
            tmpat = atomtypes.copy()
            tmpat[per] = "X"
            variants.append(tmpat)
        return variants

    def get_parameters(self, term, atomtypes):
        from itertools import permutations

        atomtypes = np.array(atomtypes)
        variants = self._get_x_variants(atomtypes)
        if term == "bonds" or term == "angles" or term == "dihedrals":
            variants += self._get_x_variants(atomtypes[::-1])
        elif term == "impropers":
            # Position 2 is the improper center
            perms = np.array([x for x in list(permutations((0, 1, 2, 3))) if x[2] == 2])
            for perm in perms:
                variants += self._get_x_variants(atomtypes[perm])
        variants = sorted(variants, key=lambda x: sum(x == "X"))

        termpar = self.prm[term]
        for var in variants:
            atomtypestr = ", ".join(var)
            if len(var) > 1:
                atomtypestr = "(" + atomtypestr + ")"
            if atomtypestr in termpar:
                return termpar[atomtypestr]
        raise RuntimeError(f"{atomtypes} doesn't have {term} information in the FF")

    def get_atom_types(self):
        return np.unique(self.prm["atomtypes"])

    def get_charge(self, at):
        params = self.get_parameters("electrostatics", [at])
        return params["charge"]

    def get_mass(self, at):
        return self.prm["masses"][at]

    def get_LJ(self, at):
        params = self.get_parameters("lj", [at])
        return params["sigma"], params["epsilon"]

    def get_bond(self, at1, at2):
        params = self.get_parameters("bonds", [at1, at2])
        return params["k0"], params["req"]

    def get_angle(self, at1, at2, at3):
        params = self.get_parameters("angles", [at1, at2, at3])
        return params["k0"], radians(params["theta0"])

    def get_dihedral(self, at1, at2, at3, at4):
        params = self.get_parameters("dihedrals", [at1, at2, at3, at4])

        terms = []
        for term in params["terms"]:
            terms.append([term["phi_k"], radians(term["phase"]), term["per"]])

        return terms

    def get_14(self, at1, at2, at3, at4):
        params = self.get_parameters("dihedrals", [at1, at2, at3, at4])

        terms = []
        for term in params["terms"]:
            terms.append([term["phi_k"], radians(term["phase"]), term["per"]])

        lj1 = self.get_parameters("lj", [at1])
        lj4 = self.get_parameters("lj", [at4])
        return (
            params["scnb"] if "scnb" in params else 1,
            params["scee"] if "scee" in params else 1,
            lj1["sigma14"],
            lj1["epsilon14"],
            lj4["sigma14"],
            lj4["epsilon14"],
        )

    def get_improper(self, at1, at2, at3, at4):
        params = self.get_parameters("impropers", [at1, at2, at3, at4])
        return params["phi_k"], radians(params["phase"]), params["per"]


# In[16]:


# ./paddlemd/forcefields/ff_parmed.py
# from paddlemd.forcefields.forcefield import _ForceFieldBase
from math import radians
import numpy as np


def load_parmed_parameters(fname):
    """ Convenience method for reading parameter files with parmed

    Parameters
    ----------
    fname : str
        Parameter file name

    Returns
    -------
    prm : ParameterSet
        A parmed ParameterSet object

    Examples
    --------
    >>> prm = loadParameters(join(home(dataDir='thrombin-ligand-amber'), 'structure.prmtop'))
    """
    import parmed

    prm = None
    if fname.endswith(".prm"):
        try:
            prm = parmed.charmm.CharmmParameterSet(fname)
        except Exception as e:
            print(
                f"Failed to read {fname} as CHARMM parameters. Attempting with AMBER prmtop reader"
            )
            try:
                struct = parmed.amber.AmberParm(fname)
                prm = parmed.amber.AmberParameterSet.from_structure(struct)
            except Exception as e2:
                print(f"Failed to read {fname} due to errors {e} {e2}")
    elif fname.endswith(".prmtop"):
        struct = parmed.amber.AmberParm(fname)
        prm = parmed.amber.AmberParameterSet.from_structure(struct)
    elif fname.endswith(".frcmod"):
        prm = parmed.amber.AmberParameterSet(fname)

    if prm is None:
        raise RuntimeError(f"Extension of file {fname} not recognized")
    return prm


class ParmedForcefield(_ForceFieldBase):
    def __init__(self, mol, prm):
        self.mol = mol
        self.prm = prm
        if isinstance(prm, str):
            self.prm = load_parmed_parameters(prm)

    def get_atom_types(self):
        return np.unique(self.mol.atomtype)

    def get_charge(self, at):
        idx = np.where(self.mol.atomtype == at)[0][0]
        return self.mol.charge[idx]

    def get_mass(self, at):
        idx = np.where(self.mol.atomtype == at)[0][0]
        return self.mol.masses[idx]

    def get_LJ(self, at):
        params = self.prm.atom_types[at]
        return params.sigma, params.epsilon

    def get_bond(self, at1, at2):
        params = self.prm.bond_types[(at1, at2)]
        return params.k, params.req

    def get_angle(self, at1, at2, at3):
        params = self.prm.angle_types[(at1, at2, at3)]
        return params.k, radians(params.theteq)

    def get_dihedral(self, at1, at2, at3, at4):
        variants = [(at1, at2, at3, at4), (at4, at3, at2, at1)]
        params = None
        for var in variants:
            if var in self.prm.dihedral_types:
                params = self.prm.dihedral_types[var]
                break

        if params is None:
            raise RuntimeError(
                f"Could not find dihedral parameters for ({at1}, {at2}, {at3}, {at4})"
            )

        terms = []
        for term in params:
            terms.append([term.phi_k, radians(term.phase), term.per])

        return terms

    def get_14(self, at1, at2, at3, at4):
        variants = [(at1, at2, at3, at4), (at4, at3, at2, at1)]
        for var in variants:
            if var in self.prm.dihedral_types:
                params = self.prm.dihedral_types[var][0]
                break

        lj1 = self.prm.atom_types[at1]
        lj4 = self.prm.atom_types[at4]
        return (
            params.scnb,
            params.scee,
            lj1.sigma_14,
            lj1.epsilon_14,
            lj4.sigma_14,
            lj4.epsilon_14,
        )

    def get_improper(self, at1, at2, at3, at4):
        from itertools import permutations

        types = np.array((at1, at2, at3, at4))
        perms = np.array([x for x in list(permutations((0, 1, 2, 3))) if x[2] == 2])
        for p in perms:
            if tuple(types[p]) in self.prm.improper_types:
                params = self.prm.improper_types[tuple(types[p])]
                return params.psi_k, radians(params.psi_eq), 0
            elif tuple(types[p]) in self.prm.improper_periodic_types:
                params = self.prm.improper_periodic_types[tuple(types[p])]
                return params.phi_k, radians(params.phase), params.per

        raise RuntimeError(f"Could not find improper parameters for key {types}")


# # parameters.py
# from paddlemd.parameters import Parameters

# In[17]:


import paddle
from math import sqrt
import numpy as np


class Parameters:
    def __init__(
        self, ff, mol, terms=None, precision=paddle.float32,
    ):
        self.A = None
        self.B = None
        self.bonds = None
        self.bond_params = None
        self.charges = None
        self.masses = None
        self.mapped_atom_types = None
        self.angles = None
        self.angle_params = None
        self.dihedrals = None
        self.dihedral_params = None
        self.idx14 = None
        self.nonbonded_14_params = None
        self.impropers = None
        self.improper_params = None

        self.natoms = mol.numAtoms
        if terms is None:
            terms = ("bonds", "angles", "dihedrals", "impropers", "1-4")
        terms = [term.lower() for term in terms]
        self.build_parameters(ff, mol, terms)
        self.precision_(precision)
        self.to_(device=None) # 为了调试步幅小，临时用None默认值代替。
 
    def to_(self, device=None):
        self.A = self.A 
        self.B = self.B 
        self.charges = self.charges 
        self.masses = self.masses 
        if self.bonds is not None:
            self.bonds = self.bonds 
            self.bond_params = self.bond_params 
        if self.angles is not None:
            self.angles = self.angles 
            self.angle_params = self.angle_params 
        if self.dihedrals is not None:
            self.dihedrals = self.dihedrals 
            for j in range(len(self.dihedral_params)):
                termparams = self.dihedral_params[j]
                termparams["idx"] = termparams["idx"] 
                termparams["params"] = termparams["params"] 
        if self.idx14 is not None:
            self.idx14 = self.idx14 
            self.nonbonded_14_params = self.nonbonded_14_params 
        if self.impropers is not None:
            self.impropers = self.impropers 
            termparams = self.improper_params[0]
            termparams["idx"] = termparams["idx"] 
            termparams["params"] = termparams["params"] 
#         self.device = device

    def precision_(self, precision):
        self.A = self.A.astype(precision)
        self.B = self.B.astype(precision)
        self.charges = self.charges.astype(precision)
        self.masses = self.masses.astype(precision)
        if self.bonds is not None:
            self.bond_params = self.bond_params.astype(precision)
        if self.angles is not None:
            self.angle_params = self.angle_params.astype(precision)
        if self.dihedrals is not None:
            for j in range(len(self.dihedral_params)):
                termparams = self.dihedral_params[j]
                termparams["params"] = termparams["params"].astype(precision)
        if self.idx14 is not None:
            self.nonbonded_14_params = self.nonbonded_14_params.astype(precision)
        if self.impropers is not None:
            termparams = self.improper_params[0]
            termparams["params"] = termparams["params"].astype(precision)

    def get_exclusions(self, types=("bonds", "angles", "1-4"), fullarray=False):
        exclusions = []
        if self.bonds is not None and "bonds" in types:
            exclusions += self.bonds.numpy().tolist()
        if self.angles is not None and "angles" in types:
            npangles = self.angles.numpy()
            exclusions += npangles[:, [0, 2]].tolist()
        if self.dihedrals is not None and "1-4" in types:
            # These exclusions will be covered by nonbonded_14_params
            npdihedrals = self.dihedrals.numpy()
            exclusions += npdihedrals[:, [0, 3]].tolist()
        if fullarray:
            fullmat = np.full((self.natoms, self.natoms), False, dtype=bool)
            if len(exclusions):
                exclusions = np.array(exclusions)
                fullmat[exclusions[:, 0], exclusions[:, 1]] = True
                fullmat[exclusions[:, 1], exclusions[:, 0]] = True
                exclusions = fullmat
        return exclusions

    def build_parameters(self, ff, mol, terms):
        uqatomtypes, indexes = np.unique(mol.atomtype, return_inverse=True)

        self.mapped_atom_types = paddle.to_tensor(indexes)
        self.charges = paddle.to_tensor(mol.charge.astype(np.float64))
        self.masses = self.make_masses(ff, mol.atomtype)
        self.A, self.B = self.make_lj(ff, uqatomtypes)
        if "bonds" in terms and len(mol.bonds):
            uqbonds = np.unique([sorted(bb) for bb in mol.bonds], axis=0)
            self.bonds = paddle.to_tensor(uqbonds.astype(np.int64))
            self.bond_params = self.make_bonds(ff, uqatomtypes[indexes[uqbonds]])
        if "angles" in terms and len(mol.angles):
            uqangles = np.unique(
                [ang if ang[0] < ang[2] else ang[::-1] for ang in mol.angles], axis=0
            )
            self.angles = paddle.to_tensor(uqangles.astype(np.int64))
            self.angle_params = self.make_angles(ff, uqatomtypes[indexes[uqangles]])
        if "dihedrals" in terms and len(mol.dihedrals):
            uqdihedrals = np.unique(
                [dih if dih[0] < dih[3] else dih[::-1] for dih in mol.dihedrals], axis=0
            )
            self.dihedrals = paddle.to_tensor(uqdihedrals.astype(np.int64))
            self.dihedral_params = self.make_dihedrals(
                ff, uqatomtypes[indexes[uqdihedrals]]
            )
        if "1-4" in terms and len(mol.dihedrals):
            # Keep only dihedrals whos 1/4 atoms are not in bond+angle exclusions
            exclusions = self.get_exclusions(types=("bonds", "angles"), fullarray=True)
            keep = ~exclusions[uqdihedrals[:, 0], uqdihedrals[:, 3]]
            dih14 = uqdihedrals[keep, :]
            if len(dih14):
                # Remove duplicates (can occur if 1,4 atoms were same and 2,3 differed)
                uq14idx = np.unique(dih14[:, [0, 3]], axis=0, return_index=True)[1]
                dih14 = dih14[uq14idx]
                self.idx14 = paddle.to_tensor(dih14[:, [0, 3]].astype(np.int64))
                self.nonbonded_14_params = self.make_14(ff, uqatomtypes[indexes[dih14]])
        if "impropers" in terms and len(mol.impropers):
            uqimpropers = np.unique(mol.impropers, axis=0)
            # uqimpropers = self._unique_impropers(mol.impropers, mol.bonds)
            self.impropers = paddle.to_tensor(uqimpropers.astype(np.int64))
            self.improper_params = self.make_impropers(
                ff, uqatomtypes, indexes, uqimpropers, uqbonds
            )

    # def make_charges(self, ff, atomtypes):
    #     return paddle.to_tensor([ff.get_charge(at) for at in atomtypes])

    def make_masses(self, ff, atomtypes):
        masses = paddle.to_tensor([ff.get_mass(at) for at in atomtypes])
        masses.unsqueeze_(1)  # natoms,1
        return masses

    def make_lj(self, ff, uqatomtypes):
        sigma = []
        epsilon = []
        for at in uqatomtypes:
            ss, ee = ff.get_LJ(at)
            sigma.append(ss)
            epsilon.append(ee)

        sigma = np.array(sigma, dtype=np.float64)
        epsilon = np.array(epsilon, dtype=np.float64)

        A, B = calculate_AB(sigma, epsilon)
        A = paddle.to_tensor(A)
        B = paddle.to_tensor(B)
        return A, B

    def make_bonds(self, ff, uqbondatomtypes):
        return paddle.to_tensor([ff.get_bond(*at) for at in uqbondatomtypes])

    def make_angles(self, ff, uqangleatomtypes):
        return paddle.to_tensor([ff.get_angle(*at) for at in uqangleatomtypes])

    def make_dihedrals(self, ff, uqdihedralatomtypes):
        from collections import defaultdict

        dihedrals = defaultdict(lambda: {"idx": [], "params": []})

        for i, at in enumerate(uqdihedralatomtypes):
            terms = ff.get_dihedral(*at)
            for j, term in enumerate(terms):
                dihedrals[j]["idx"].append(i)
                dihedrals[j]["params"].append(term)

        maxterms = max(dihedrals.keys()) + 1
        newdihedrals = []
        for j in range(maxterms):
            dihedrals[j]["idx"] = paddle.to_tensor(dihedrals[j]["idx"])
            dihedrals[j]["params"] = paddle.to_tensor(dihedrals[j]["params"])
            newdihedrals.append(dihedrals[j])

        return newdihedrals

    def make_impropers(self, ff, uqatomtypes, indexes, uqimpropers, bonds):
        impropers = {"idx": [], "params": []}
        graph = improper_graph(uqimpropers, bonds)

        for i, impr in enumerate(uqimpropers):
            at = uqatomtypes[indexes[impr]]
            try:
                params = ff.get_improper(*at)
            except:
                center = detect_improper_center(impr, graph)
                notcenter = sorted(np.setdiff1d(impr, center))
                order = [notcenter[0], notcenter[1], center, notcenter[2]]
                at = uqatomtypes[indexes[order]]
                params = ff.get_improper(*at)

            impropers["idx"].append(i)
            impropers["params"].append(params)

        impropers["idx"] = paddle.to_tensor(impropers["idx"])
        impropers["params"] = paddle.to_tensor(impropers["params"])
        return [impropers]

    def make_14(self, ff, uq14atomtypes):
        nonbonded_14_params = []
        for uqdih in uq14atomtypes:
            scnb, scee, lj1_s14, lj1_e14, lj4_s14, lj4_e14 = ff.get_14(*uqdih)
            # Lorentz - Berthelot combination rule
            sig = 0.5 * (lj1_s14 + lj4_s14)
            eps = sqrt(lj1_e14 * lj4_e14)
            s6 = sig ** 6
            s12 = s6 * s6
            A = eps * 4 * s12
            B = eps * 4 * s6
            nonbonded_14_params.append([A, B, scnb, scee])
        return paddle.to_tensor(nonbonded_14_params)


def calculate_AB(sigma, epsilon):
    # Lorentz - Berthelot combination rule
    sigma_table = 0.5 * (sigma + sigma[:, None])
    eps_table = np.sqrt(epsilon * epsilon[:, None])
    sigma_table_6 = sigma_table ** 6
    sigma_table_12 = sigma_table_6 * sigma_table_6
    A = eps_table * 4 * sigma_table_12
    B = eps_table * 4 * sigma_table_6
    del sigma_table_12, sigma_table_6, eps_table, sigma_table
    return A, B


def detect_improper_center(indexes, graph):
    for i in indexes:
        if len(np.intersect1d(list(graph.neighbors(i)), indexes)) == 3:
            return i


def improper_graph(impropers, bonds):
    import networkx as nx

    g = nx.Graph()
    g.add_nodes_from(np.unique(impropers))
    g.add_edges_from([tuple(b) for b in bonds])
    return g


# # integrator.py

# In[18]:


import numpy as np
import paddle

TIMEFACTOR = 48.88821
BOLTZMAN = 0.001987191

def paddlerandn_like(x) : # 添加飞桨的randn_like函数
    return paddle.randn(x.shape)

def kinetic_energy(masses, vel):
    Ekin = paddle.sum(0.5 * paddle.sum(vel * vel, axis=2, keepdim=True) * masses, axis=1)
    return Ekin


def maxwell_boltzmann(masses, T, replicas=1):
    natoms = len(masses)
    velocities = []
    for i in range(replicas):
        velocities.append(
            paddle.sqrt(T * BOLTZMAN / masses) * paddle.randn((natoms, 3)).astype(masses.dtype)
        )

    return paddle.stack(velocities, axis=0)


def kinetic_to_temp(Ekin, natoms):
    return 2.0 / (3.0 * natoms * BOLTZMAN) * Ekin


def _first_VV(pos, vel, force, mass, dt):
    accel = force / mass
    pos += vel * dt + 0.5 * accel * dt * dt
    vel += 0.5 * dt * accel


def _second_VV(vel, force, mass, dt):
    accel = force / mass
    vel += 0.5 * dt * accel


def langevin(vel, gamma, coeff, dt): 
#     csi = paddle.randn_like(vel, device=device) * coeff
    csi = paddlerandn_like(vel) * coeff
    vel += -gamma * vel * dt + csi


PICOSEC2TIMEU = 1000.0 / TIMEFACTOR


class Integrator:
    def __init__(self, systems, forces, timestep, device=None, gamma=None, T=None): # 临时用device=None
        self.dt = timestep / TIMEFACTOR
        self.systems = systems
        self.forces = forces
#         self.device = device
        gamma = gamma / PICOSEC2TIMEU
        self.gamma = gamma
        self.T = T
        if T:
            M = self.forces.par.masses
            self.vcoeff = paddle.sqrt(2.0 * gamma / M * BOLTZMAN * T * self.dt)

    def step(self, niter=1):
        s = self.systems
        masses = self.forces.par.masses
        natoms = len(masses)
        with Benchmark("Integrator.step time"):
            for _ in range(niter):
                _first_VV(s.pos, s.vel, s.forces, masses, self.dt)
                pot = self.forces.compute(s.pos, s.box, s.forces)
                if self.T:
                    langevin(s.vel, self.gamma, self.vcoeff, self.dt)
                _second_VV(s.vel, s.forces, masses, self.dt)

        Ekin = np.array([v.item() for v in kinetic_energy(masses, s.vel)])
        T = kinetic_to_temp(Ekin, natoms)
        return Ekin, pot, T


# # systems.py

# In[19]:


import paddle
import numpy as np


class System:
    def __init__(self, natoms, nreplicas, precision):
        # self.pos = pos  # Nsystems,Natoms,3
        # self.vel = vel  # Nsystems,Natoms,3
        # self.box = box
        # self.forces = forces
        self.box = paddle.zeros([nreplicas, 3, 3])
        self.pos = paddle.zeros([nreplicas, natoms, 3])
        self.vel = paddle.zeros([nreplicas, natoms, 3])
        self.forces = paddle.zeros([nreplicas, natoms, 3])

#         self.to_(device)
        self.precision_(precision)

    @property
    def natoms(self):
        return self.pos.shape[1]

    @property
    def nreplicas(self):
        return self.pos.shape[0]

#     def to_(self, device):
#         self.forces = self.forces 
#         self.box = self.box 
#         self.pos = self.pos 
#         self.vel = self.vel 

    def precision_(self, precision):
        self.forces = self.forces.astype(precision)
        self.box = self.box.astype(precision)
        self.pos = self.pos.astype(precision)
        self.vel = self.vel.astype(precision)

    def set_positions(self, pos):
        if pos.shape[1] != 3:
            raise RuntimeError(
                "Positions shape must be (natoms, 3, 1) or (natoms, 3, nreplicas)"
            )

        atom_pos = np.transpose(pos, (2, 0, 1))
        if self.nreplicas > 1 and atom_pos.shape[0] != self.nreplicas:
            atom_pos = np.repeat(atom_pos[0][None, :], self.nreplicas, axis=0)

        self.pos[:] = paddle.to_tensor(
            atom_pos, dtype=self.pos.dtype)

    def set_velocities(self, vel):
        if vel.shape != [self.nreplicas, self.natoms, 3]:
            raise RuntimeError("Velocities shape must be (nreplicas, natoms, 3)")
        self.vel[:] = vel.clone().detach().astype(self.vel.dtype)

    def set_box(self, box):
        if box.ndim == 1:
            if len(box) != 3:
                raise RuntimeError("Box must have at least 3 elements")
            box = box[:, None]

        if box.shape[0] != 3:
            raise RuntimeError("Box shape must be (3, 1) or (3, nreplicas)")

        box = np.swapaxes(box, 1, 0)

        if self.nreplicas > 1 and box.shape[0] != self.nreplicas:
            box = np.repeat(box[0][None, :], self.nreplicas, axis=0)

        for r in range(box.shape[0]):
#             self.box[r][paddle.eye(3).astype(paddle.bool)] = paddle.to_tensor(
#                 box[r], dtype=self.box.dtype)
            self.box[r] = paddle.to_tensor(
                box[r], dtype=self.box.dtype) * paddle.eye(3).astype(paddle.bool)
            

    def set_forces(self, forces):
        if forces.shape != [self.nreplicas, self.natoms, 3]:
            raise RuntimeError("Forces shape must be (nreplicas, natoms, 3)")
        self.forces[:] = paddle.to_tensor(
            forces, dtype=self.forces.dtype)


# # forces.py

# In[20]:


from scipy import constants as const
import paddle
import numpy as np
from math import pi

import paddle
import math
# 发现飞桨支持atan2函数，且自己写的只适合1D数据
# def paddleatan2(input, other):
#     atan = paddle.atan(input/other)
#     atan[1] = atan[1] + pi
#     atan[2] = atan[2] + pi
#     return atan

def paddlescatter(x, dim, index, src): # 支持1D版本
    
    updates = src
    if len(index.shape) == 1 :
#         for i in index:
#             x[i] += updates[i]
        for i in range(index.shape[0]):
            x[index[i]] += updates[i]
        return x
                                
    i, j = index.shape
    grid_x , grid_y = paddle.meshgrid(paddle.arange(i), paddle.arange(j))
    if dim == 0 :
        index = paddle.stack([index.flatten(), grid_y.flatten()], axis=1)
    elif dim == 1:
        index = paddle.stack([grid_x.flatten(), index.flatten()], axis=1)
        
    # PaddlePaddle updates 的 shape 大小必须与 index 对应
    updates_index = paddle.stack([grid_x.flatten(), grid_y.flatten()], axis=1)
    updates = paddle.gather_nd(updates, index=updates_index)
    return paddle.scatter_nd_add(x, index, updates)

def paddleindex_add(x, dim, index, source): # 飞桨的index_add
    for i in range(len(index)):
        x[index[i]] += source[i]
    return x

def paddleeye(x, n):
    tmp =x[0][paddle.eye(n).astype(paddle.bool)]
    return tmp.unsqueeze_(0)

class Forces:
    """
    Parameters
    ----------
    cutoff : float
        If set to a value it will only calculate LJ, electrostatics and bond energies for atoms which are closer
        than the threshold
    rfa : bool
        Use with `cutoff` to enable the reaction field approximation for scaling of the electrostatics up to the cutoff.
        Uses the value of `solventDielectric` to model everything beyond the cutoff distance as solvent with uniform
        dielectric.
    solventDielectric : float
        Used together with `cutoff` and `rfa`
    """

    # 1-4 is nonbonded but we put it currently in bonded to not calculate all distances
    bonded = ["bonds", "angles", "dihedrals", "impropers", "1-4"]
    nonbonded = ["electrostatics", "lj", "repulsion", "repulsioncg"]
    terms = bonded + nonbonded

    def __init__(
        self,
        parameters,
        terms=None,
        external=None,
        cutoff=None,
        rfa=False,
        solventDielectric=78.5,
        switch_dist=None,
        exclusions=("bonds", "angles", "1-4"),
    ):
        self.par = parameters
        if terms is None:
            raise RuntimeError(
                'Set force terms or leave empty brackets [].\nAvailable options: "bonds", "angles", "dihedrals", "impropers", "1-4", "electrostatics", "lj", "repulsion", "repulsioncg".'
            )

        self.energies = [ene.lower() for ene in terms]
        for et in self.energies:
            if et not in Forces.terms:
                raise ValueError(f"Force term {et} is not implemented.")

        if "1-4" in self.energies and "dihedrals" not in self.energies:
            raise RuntimeError(
                "You cannot enable 1-4 interactions without enabling dihedrals"
            )

        self.natoms = len(parameters.masses)
        self.require_distances = any(f in self.nonbonded for f in self.energies)
        self.ava_idx = (
            self._make_indeces(
                self.natoms, parameters.get_exclusions(exclusions))
            if self.require_distances
            else None
        )
        self.external = external
        self.cutoff = cutoff
        self.rfa = rfa
        self.solventDielectric = solventDielectric
        self.switch_dist = switch_dist

    def _filter_by_cutoff(self, dist, arrays):
        under_cutoff = dist <= self.cutoff
        indexedarrays = []
        for arr in arrays:
            indexedarrays.append(arr[under_cutoff])
        return indexedarrays

    def compute(self, pos, box, forces, returnDetails=False, explicit_forces=True):
        if not explicit_forces and not pos.requires_grad:
            raise RuntimeError(
                "The positions passed don't require gradients. Please use pos.detach().requires_grad_(True) before passing."
            )

        nsystems = pos.shape[0]
        if paddle.any(paddle.isnan(pos)):
            raise RuntimeError("Found NaN coordinates.")

        pot = []
        print(f"==start forces compute...")
        with Benchmark("for i in range(nsystems)耗时:"):
            for i in range(nsystems):
                pp = {
                    v: paddle.zeros([1]).astype(pos.dtype)
                    for v in self.energies
                }
                pp["external"] = paddle.zeros([1]).astype(pos.dtype)
                pot.append(pp)

        forces.zero_()
        for i in range(nsystems):
            spos = pos[i] # pos[1,688,3] spos[688,3] 
#             sbox = box[i][paddle.eye(3).astype(paddle.bool)]  # Use only the diagonal
            sbox = paddleeye(box, 3) # 将tensor eye转为自定义函数 box[1,3,3] sbox[3]
            print(f"==sbox = box[i][torch.eye(3).bool()] sbox.shape:{sbox.shape}  box[i].shape:{ box[i].shape} box.shape:{box.shape} pos.shape:{pos.shape} spos.shape:{spos.shape}")
#             print(f"sbos, box shape {sbox.shape, box.shape}")
            # Bonded terms
            # TODO: We are for sure doing duplicate distance calculations here!
            if "bonds" in self.energies and self.par.bonds is not None: # 键
                bond_dist, bond_unitvec, _ = calculate_distances(
                    spos, self.par.bonds, sbox
                )
                pairs = self.par.bonds
                bond_params = self.par.bond_params
                if self.cutoff is not None:
                    (
                        bond_dist,
                        bond_unitvec,
                        pairs,
                        bond_params,
                    ) = self._filter_by_cutoff(
                        bond_dist, (bond_dist, bond_unitvec, pairs, bond_params)
                    )
                E, force_coeff = evaluate_bonds(bond_dist, bond_params, explicit_forces)

#                 pot[i]["bonds"] += E.sum() # 二阶赋值都要加上中间量来实现
                tmp = pot[i]
                tmp["bonds"] += E.sum()
                pot[i] = tmp
                if explicit_forces:
                    forcevec = bond_unitvec * force_coeff[:, None]
                    forces[i] = paddleindex_add(forces[i], 0, pairs[:, 0], -forcevec)
                    forces[i] = paddleindex_add(forces[i], 0, pairs[:, 1], forcevec)

            if "angles" in self.energies and self.par.angles is not None: # 角度
                _, _, r21 = calculate_distances(spos, self.par.angles[:, 0:2], sbox)
                _, _, r23 = calculate_distances(spos, self.par.angles[:, 2:0:-1], sbox)
                E, angle_forces = evaluate_angles(
                    r21, r23, self.par.angle_params, explicit_forces
                )

#                 pot[i]["angles"] += E.sum()
                tmp = pot[i]
                tmp["angles"] += E.sum()
                pot[i] = tmp
                if explicit_forces:
                    forces[i] = paddleindex_add(forces[i], 0, self.par.angles[:, 0], angle_forces[0])
                    forces[i] = paddleindex_add(forces[i], 0, self.par.angles[:, 1], angle_forces[1])
                    forces[i] = paddleindex_add(forces[i], 0, self.par.angles[:, 2], angle_forces[2])

            if "dihedrals" in self.energies and self.par.dihedrals is not None: # 二面角
#                 print(f"== spos, sbox {spos, sbox} self.par.dihedrals {self.par.dihedrals}")
#                 print(f"==_, _, r12 = calculate_distances {spos, self.par.dihedrals[:, 0:2], sbox}")
                _, _, r12 = calculate_distances(
                    spos, self.par.dihedrals[:, 0:2], sbox
                )
                _, _, r23 = calculate_distances(
                    spos, self.par.dihedrals[:, 1:3], sbox
                )
                _, _, r34 = calculate_distances(
                    spos, self.par.dihedrals[:, 2:4], sbox
                )
                E, dihedral_forces = evaluate_torsion(
                    r12, r23, r34, self.par.dihedral_params, explicit_forces
                )

#                 pot[i]["dihedrals"] += E.sum()
                tmp = pot[i]
                tmp["dihedrals"] += E.sum()
                pot[i] = tmp
                if explicit_forces:
                    forces[i] = paddleindex_add(forces[i], 
                        0, self.par.dihedrals[:, 0], dihedral_forces[0]
                    )
                    forces[i] = paddleindex_add(forces[i], 
                        0, self.par.dihedrals[:, 1], dihedral_forces[1]
                    )
                    forces[i] = paddleindex_add(forces[i], 
                        0, self.par.dihedrals[:, 2], dihedral_forces[2]
                    )
                    forces[i] = paddleindex_add(forces[i], 
                        0, self.par.dihedrals[:, 3], dihedral_forces[3]
                    )

            if "1-4" in self.energies and self.par.idx14 is not None:
                nb_dist, nb_unitvec, _ = calculate_distances(spos, self.par.idx14, sbox)

                nonbonded_14_params = self.par.nonbonded_14_params
                idx14 = self.par.idx14
                # if self.cutoff is not None:
                #     (
                #         nb_dist,
                #         nb_unitvec,
                #         nonbonded_14_params,
                #         idx14,
                #     ) = self._filter_by_cutoff(
                #         nb_dist,
                #         (
                #             nb_dist,
                #             nb_unitvec,
                #             self.par.nonbonded_14_params,
                #             self.par.idx14,
                #         ),
                #     )

                aa = nonbonded_14_params[:, 0]
                bb = nonbonded_14_params[:, 1]
                scnb = nonbonded_14_params[:, 2]
                scee = nonbonded_14_params[:, 3]

                if "lj" in self.energies: # LJ系数 Lennard-Jones
                    E, force_coeff = evaluate_LJ_internal(
                        nb_dist, aa, bb, scnb, None, None, explicit_forces
                    )
#                     pot[i]["lj"] += E.sum()
                    tmp = pot[i]
                    tmp["lj"] += E.sum()
                    pot[i] = tmp
                    if explicit_forces:
                        forcevec = nb_unitvec * force_coeff[:, None]
                        forces[i] = paddleindex_add(forces[i], 0, idx14[:, 0], -forcevec)
                        forces[i] = paddleindex_add(forces[i], 0, idx14[:, 1], forcevec)
                if "electrostatics" in self.energies: # 静力场
                    E, force_coeff = evaluate_electrostatics(
                        nb_dist,
                        idx14,
                        self.par.charges,
                        scee,
                        cutoff=None,
                        rfa=False,
                        solventDielectric=self.solventDielectric,
                        explicit_forces=explicit_forces,
                    )
#                     pot[i]["electrostatics"] += E.sum()
                    tmp = pot[i]
                    tmp["electrostatics"] += E.sum()
                    pot[i] = tmp
                    if explicit_forces:
                        forcevec = nb_unitvec * force_coeff[:, None]
                        forces[i] = paddleindex_add(forces[i], 0, idx14[:, 0], -forcevec)
                        forces[i] = paddleindex_add(forces[i], 0, idx14[:, 1], forcevec)

            if "impropers" in self.energies and self.par.impropers is not None:
                _, _, r12 = calculate_distances(
                    spos, self.par.impropers[:, 0:2], sbox
                )
                _, _, r23 = calculate_distances(
                    spos, self.par.impropers[:, 1:3], sbox
                )
                _, _, r34 = calculate_distances(
                    spos, self.par.impropers[:, 2:4], sbox
                )
                E, improper_forces = evaluate_torsion(
                    r12, r23, r34, self.par.improper_params, explicit_forces
                )

#                 pot[i]["impropers"] += E.sum()
                tmp = pot[i]
                tmp["impropers"] += E.sum()
                pot[i] = tmp
                if explicit_forces:
                    forces[i] = paddleindex_add(forces[i], 
                        0, self.par.impropers[:, 0], improper_forces[0]
                    )
                    forces[i] = paddleindex_add(forces[i], 
                        0, self.par.impropers[:, 1], improper_forces[1]
                    )
                    forces[i] = paddleindex_add(forces[i], 
                        0, self.par.impropers[:, 2], improper_forces[2]
                    )
                    forces[i] = paddleindex_add(forces[i], 
                        0, self.par.impropers[:, 3], improper_forces[3]
                    )

            # Non-bonded terms
            if self.require_distances and len(self.ava_idx):
                # Lazy mode: Do all vs all distances
                # TODO: These distance calculations are fucked once we do neighbourlists since they will vary per system!!!!
                nb_dist, nb_unitvec, _ = calculate_distances(spos, self.ava_idx, sbox)
                ava_idx = self.ava_idx
                if self.cutoff is not None:
                    nb_dist, nb_unitvec, ava_idx = self._filter_by_cutoff(
                        nb_dist, (nb_dist, nb_unitvec, ava_idx)
                    )

                for v in self.energies:
                    if v == "electrostatics":
                        E, force_coeff = evaluate_electrostatics(
                            nb_dist,
                            ava_idx,
                            self.par.charges,
                            cutoff=self.cutoff,
                            rfa=self.rfa,
                            solventDielectric=self.solventDielectric,
                            explicit_forces=explicit_forces,
                        )
#                         pot[i][v] += E.sum()
                        tmp = pot[i]
                        tmp[v] += E.sum()
                        pot[i] = tmp
                    elif v == "lj":
                        E, force_coeff = evaluate_LJ(
                            nb_dist,
                            ava_idx,
                            self.par.mapped_atom_types,
                            self.par.A,
                            self.par.B,
                            self.switch_dist,
                            self.cutoff,
                            explicit_forces,
                        )
#                         pot[i][v] += E.sum()
                        tmp = pot[i]
                        tmp[v] += E.sum()
                        pot[i] = tmp
                    elif v == "repulsion":
                        E, force_coeff = evaluate_repulsion(
                            nb_dist,
                            ava_idx,
                            self.par.mapped_atom_types,
                            self.par.A,
                            explicit_forces,
                        )
#                         pot[i][v] += E.sum()
                        tmp = pot[i]
                        tmp[v] += E.sum()
                        pot[i] = tmp
                    elif v == "repulsioncg":
                        E, force_coeff = evaluate_repulsion_CG(
                            nb_dist,
                            ava_idx,
                            self.par.mapped_atom_types,
                            self.par.B,
                            explicit_forces,
                        )
#                         pot[i][v] += E.sum()
                        tmp = pot[i]
                        tmp[v] += E.sum()
                        pot[i] = tmp
                    else:
                        continue

                    if explicit_forces:
                        forcevec = nb_unitvec * force_coeff[:, None]
                        forces[i] = paddleindex_add(forces[i], 0, ava_idx[:, 0], -forcevec)
                        forces[i] = paddleindex_add(forces[i], 0, ava_idx[:, 1], forcevec)

        if self.external:
            ext_ene, ext_force = self.external.calculate(pos, box)
            for s in range(nsystems):
                pot[s]["external"] += ext_ene[s]
            if explicit_forces:
                forces += ext_force

        if not explicit_forces:
            enesum = paddle.zeros([1], dtype=pos.dtype)
            for i in range(nsystems):
                for ene in pot[i]:
                    if pot[i][ene].requires_grad:
                        enesum += pot[i][ene]
            forces[:] = -paddle.autograd.grad(
                enesum, pos, only_inputs=True, retain_graph=True
            )[0]
            if returnDetails:
                return pot
            else:
                return [paddle.sum(paddle.cat(list(pp.values()))) for pp in pot]

        if returnDetails:
#             return [{k: v.cpu().item() for k, v in pp.items()} for pp in pot]
            return [{k: v.item() for k, v in pp.items()} for pp in pot]
        else:
#             return [np.sum([v.cpu().item() for _, v in pp.items()]) for pp in pot]
            return [np.sum([v.item() for _, v in pp.items()]) for pp in pot]

    def _make_indeces(self, natoms, excludepairs):
        fullmat = np.full((natoms, natoms), True, dtype=bool)
        if len(excludepairs):
            excludepairs = np.array(excludepairs)
            fullmat[excludepairs[:, 0], excludepairs[:, 1]] = False
            fullmat[excludepairs[:, 1], excludepairs[:, 0]] = False
        fullmat = np.triu(fullmat, +1)
        allvsall_indeces = np.vstack(np.where(fullmat)).T
        ava_idx = paddle.to_tensor(allvsall_indeces) 
        return ava_idx


def wrap_dist(dist, box):
    if box is None or paddle.all(box == 0):
        wdist = dist
    else:
        wdist = dist - box.unsqueeze(0) * paddle.round(dist / box.unsqueeze(0))
    return wdist


def calculate_distances(atom_pos, atom_idx, box):
#     print(f"==calculate_distances {atom_pos, atom_idx, box}")
    direction_vec = wrap_dist(atom_pos[atom_idx[:, 0]] - atom_pos[atom_idx[:, 1]], box)
    dist = paddle.norm(direction_vec, axis=1)
    direction_unitvec = direction_vec / dist.unsqueeze(1)
    return dist, direction_unitvec, direction_vec


ELEC_FACTOR = 1 / (4 * const.pi * const.epsilon_0)  # Coulomb's constant
ELEC_FACTOR *= const.elementary_charge ** 2  # Convert elementary charges to Coulombs
ELEC_FACTOR /= const.angstrom  # Convert Angstroms to meters
ELEC_FACTOR *= const.Avogadro / (const.kilo * const.calorie)  # Convert J to kcal/mol


def evaluate_LJ(
    dist, pair_indeces, atom_types, A, B, switch_dist, cutoff, explicit_forces=True
):
    atomtype_indices = atom_types[pair_indeces]
    aa = A[atomtype_indices[:, 0], atomtype_indices[:, 1]]
    bb = B[atomtype_indices[:, 0], atomtype_indices[:, 1]]
    return evaluate_LJ_internal(dist, aa, bb, 1, switch_dist, cutoff, explicit_forces)


def evaluate_LJ_internal(
    dist, aa, bb, scale, switch_dist, cutoff, explicit_forces=True
):
    force = None

    rinv1 = 1 / dist
    rinv6 = rinv1 ** 6
    rinv12 = rinv6 * rinv6

    pot = ((aa * rinv12) - (bb * rinv6)) / scale
    if explicit_forces:
        force = (-12 * aa * rinv12 + 6 * bb * rinv6) * rinv1 / scale

    # Switching function
    if switch_dist is not None and cutoff is not None:
        mask = dist > switch_dist
        t = (dist[mask] - switch_dist) / (cutoff - switch_dist)
        switch_val = 1 + t * t * t * (-10 + t * (15 - t * 6))
        if explicit_forces:
            switch_deriv = t * t * (-30 + t * (60 - t * 30)) / (cutoff - switch_dist)
            force[mask] = (
                switch_val * force[mask] + pot[mask] * switch_deriv / dist[mask]
            )
        pot[mask] = pot[mask] * switch_val

    return pot, force


def evaluate_repulsion(
    dist, pair_indeces, atom_types, A, scale=1, explicit_forces=True
):  # LJ without B
    force = None

    atomtype_indices = atom_types[pair_indeces]
    aa = A[atomtype_indices[:, 0], atomtype_indices[:, 1]]

    rinv1 = 1 / dist
    rinv6 = rinv1 ** 6
    rinv12 = rinv6 * rinv6

    pot = (aa * rinv12) / scale
    if explicit_forces:
        force = (-12 * aa * rinv12) * rinv1 / scale
    return pot, force


def evaluate_repulsion_CG(
    dist, pair_indeces, atom_types, B, scale=1, explicit_forces=True
):  # Repulsion like from CGNet
    force = None

    atomtype_indices = atom_types[pair_indeces]
    coef = B[atomtype_indices[:, 0], atomtype_indices[:, 1]]

    rinv1 = 1 / dist
    rinv6 = rinv1 ** 6

    pot = (coef * rinv6) / scale
    if explicit_forces:
        force = (-6 * coef * rinv6) * rinv1 / scale
    return pot, force


def evaluate_electrostatics(
    dist,
    pair_indeces,
    atom_charges,
    scale=1,
    cutoff=None,
    rfa=False,
    solventDielectric=78.5,
    explicit_forces=True,
):
    force = None
    if rfa:  # Reaction field approximation for electrostatics with cutoff
        # http://docs.openmm.org/latest/userguide/theory.html#coulomb-interaction-with-cutoff
        # Ilario G. Tironi, René Sperb, Paul E. Smith, and Wilfred F. van Gunsteren. A generalized reaction field method
        # for molecular dynamics simulations. Journal of Chemical Physics, 102(13):5451–5459, 1995.
        denom = (2 * solventDielectric) + 1
        krf = (1 / cutoff ** 3) * (solventDielectric - 1) / denom
        crf = (1 / cutoff) * (3 * solventDielectric) / denom
        common = (
            ELEC_FACTOR
            * atom_charges[pair_indeces[:, 0]]
            * atom_charges[pair_indeces[:, 1]]
            / scale
        )
        dist2 = dist ** 2
        pot = common * ((1 / dist) + krf * dist2 - crf)
        if explicit_forces:
            force = common * (2 * krf * dist - 1 / dist2)
    else:
        pot = (
            ELEC_FACTOR
            * atom_charges[pair_indeces[:, 0]]
            * atom_charges[pair_indeces[:, 1]]
            / dist
            / scale
        )
        if explicit_forces:
            force = -pot / dist
    return pot, force


def evaluate_bonds(dist, bond_params, explicit_forces=True):
    force = None

    k0 = bond_params[:, 0]
    d0 = bond_params[:, 1]
    x = dist - d0
    pot = k0 * (x ** 2)
    if explicit_forces:
        force = 2 * k0 * x
    return pot, force


def evaluate_angles(r21, r23, angle_params, explicit_forces=True):
    k0 = angle_params[:, 0]
    theta0 = angle_params[:, 1]

    dotprod = paddle.sum(r23 * r21, axis=1)
    norm23inv = 1 / paddle.norm(r23, axis=1)
    norm21inv = 1 / paddle.norm(r21, axis=1)

    cos_theta = dotprod * norm21inv * norm23inv
    cos_theta = paddle.clip(cos_theta, -1, 1)
    theta = paddle.acos(cos_theta)

    delta_theta = theta - theta0
    pot = k0 * delta_theta * delta_theta

    force0, force1, force2 = None, None, None
    if explicit_forces:
        sin_theta = paddle.sqrt(1.0 - cos_theta * cos_theta)
        coef = paddle.zeros_like(sin_theta)
        nonzero = sin_theta != 0
        coef[nonzero] = -2.0 * k0[nonzero] * delta_theta[nonzero] / sin_theta[nonzero]
        force0 = (
            coef[:, None]
            * (cos_theta[:, None] * r21 * norm21inv[:, None] - r23 * norm23inv[:, None])
            * norm21inv[:, None]
        )
        force2 = (
            coef[:, None]
            * (cos_theta[:, None] * r23 * norm23inv[:, None] - r21 * norm21inv[:, None])
            * norm23inv[:, None]
        )
        force1 = -(force0 + force2)

    return pot, (force0, force1, force2)


def evaluate_torsion(r12, r23, r34, torsion_params, explicit_forces=True):
    # Calculate dihedral angles from vectors
    crossA = paddle.cross(r12, r23, axis=1)
    crossB = paddle.cross(r23, r34, axis=1)
    crossC = paddle.cross(r23, crossA, axis=1)
    normA = paddle.norm(crossA, axis=1)
    normB = paddle.norm(crossB, axis=1)
    normC = paddle.norm(crossC, axis=1)
    normcrossB = crossB / normB.unsqueeze(1)
    cosPhi = paddle.sum(crossA * normcrossB, axis=1) / normA
    sinPhi = paddle.sum(crossC * normcrossB, axis=1) / normC
    phi = -paddle.atan2(sinPhi, cosPhi)

    ntorsions = len(torsion_params[0]["idx"])
#     pot = paddle.zeros(ntorsions, dtype=r12.dtype, layout=r12.layout)
    pot = paddle.zeros([ntorsions], dtype=r12.dtype) # 飞桨无layout参数
    if explicit_forces:
#         coeff = paddle.zeros(
#             [ntorsions], dtype=r12.dtype)
        coeff = paddle.zeros([ntorsions], dtype=r12.dtype)
    for i in range(0, len(torsion_params)):
        idx = torsion_params[i]["idx"]
        k0 = torsion_params[i]["params"][:, 0]
        phi0 = torsion_params[i]["params"][:, 1]
        per = torsion_params[i]["params"][:, 2]

        if paddle.all(per > 0):  # AMBER torsions
            angleDiff = per * phi[idx] - phi0
#             pot.scatter_add_(0, idx, k0 * (1 + paddle.cos(angleDiff)))
#             print(f"==src {(k0 * (1 + paddle.cos(angleDiff))).shape}")
#             print(f"==x=pot, , index=idx{pot.shape, idx.shape} src {(k0 * (1 + paddle.cos(angleDiff))).shape}")
            pot = paddlescatter(x=pot, dim=0, index=idx, src=k0 * (1 + paddle.cos(angleDiff))) # x, dim, index, src
#             print(f"==after pot.shape{pot.shape}")
            if explicit_forces:
#                 coeff.scatter_add_(0, idx, -per * k0 * paddle.sin(angleDiff))
                coeff = paddlescatter(coeff, 0, idx, -per * k0 * paddle.sin(angleDiff))
        else:  # CHARMM torsions
            angleDiff = phi[idx] - phi0
            angleDiff[angleDiff < -pi] = angleDiff[angleDiff < -pi] + 2 * pi
            angleDiff[angleDiff > pi] = angleDiff[angleDiff > pi] - 2 * pi
#             pot.scatter_add_(0, idx, k0 * angleDiff ** 2)
            pot = paddlescatter(pot, 0, idx, k0 * angleDiff ** 2)
            if explicit_forces:
#                 coeff.scatter_add_(0, idx, 2 * k0 * angleDiff)
                coeff = paddlescatter(coeff, 0, idx, 2 * k0 * angleDiff)

    # coeff.unsqueeze_(1)

    force0, force1, force2, force3 = None, None, None, None
    if explicit_forces:
        # Taken from OpenMM
        normDelta2 = paddle.norm(r23, axis=1)
        norm2Delta2 = normDelta2 ** 2
        forceFactor0 = (-coeff * normDelta2) / (normA ** 2)
        forceFactor1 = paddle.sum(r12 * r23, axis=1) / norm2Delta2
        forceFactor2 = paddle.sum(r34 * r23, axis=1) / norm2Delta2
        forceFactor3 = (coeff * normDelta2) / (normB ** 2)

        force0vec = forceFactor0.unsqueeze(1) * crossA
        force3vec = forceFactor3.unsqueeze(1) * crossB
        s = (
            forceFactor1.unsqueeze(1) * force0vec
            - forceFactor2.unsqueeze(1) * force3vec
        )

        force0 = -force0vec
        force1 = force0vec + s
        force2 = force3vec - s
        force3 = -force3vec

    return pot, (force0, force1, force2, force3)


# # wrapper.py

# In[21]:


import paddle

def paddleeye(x, n):
    tmp =x[0][paddle.eye(n).astype(paddle.bool)]
    return tmp.unsqueeze_(0)

def paddleindexjia (x, y, xindex):
    '''
    切片+索引，使用循环来解决切片问题，然后使用中间变量，来实现按照索引赋值
    支持类似的语句pos[:, group] -= offset.unsqueeze(1)
    '''
    xlen = len(x)
    assert len(x.shape) == 3 , "维度不一致,必须为3D数据"
#     if len(y.shape) == 3 and y.shape[0] ==1 :
#         y = paddle.squeeze(y)
    assert len(y.shape) ==2 , "维度不一致，必须为2D数据"
    for i in range(xlen):
        tmp = x[i]
        tmp[xindex] += y
        x[i] = tmp
    return x


class Wrapper:
    def __init__(self, natoms, bonds):
        self.groups, self.nongrouped = calculate_molecule_groups(natoms, bonds)
        # self.groups [22] self.nongrouped 688个[3]
#         print(f"==self.groups, self.nongrouped {self.groups, self.nongrouped}") 

    def wrap(self, pos, box, wrapidx=None):
        nmol = len(self.groups)
#         print(f"== box.sahpe {box.shape}")
#         box = box[:, paddle.eye(3).astype(paddle.bool)]  # Use only the diagonal
#         box = box[:][paddle.eye(3).astype(paddle.bool)]
#         box = box[paddle.eye(3).astype(paddle.bool)]
#         box = box.reshape([3, 3]) # 先试试这样的shape可以不？ 速度15
#         box = box* (paddle.eye(3).astype(paddle.bool))
#         print(f"== after eye box.sahpe {box.shape}")
#         box = box.reshape([-1, 3, 3])
#         box[0] = box[0] * (paddle.eye(3).astype(paddle.bool)) # 速度15 torch速度9 
        box = paddleeye(box, 3)
        if paddle.all(box == 0):
            return

        if wrapidx is not None:
            # Get COM of wrapping center group
#             com = paddle.sum(pos[:, wrapidx], axis=1) / len(wrapidx)
            com = paddle.sum(paddle.gather(pos, wrapidx, axis=1), axis=1) / len(wrapidx)
            # Subtract COM from all atoms so that the center mol is at [box/2, box/2, box/2]
            pos = (pos - com) + (box / 2)

        if nmol != 0:
            # Work out the COMs and offsets of every group and move group to [0, box] range
            for i, group in enumerate(self.groups):
#                 print(f"==i, group {i, group}")
#                 tmp_com = paddle.sum(pos[:, group], axis=1) / len(group)
                tmp_com = paddle.sum(paddle.gather(pos, group, axis=1), axis=1) / len(group)
                offset = paddle.floor(tmp_com / box) * box
#                 print(f"pos group offset {pos.shape, offset.shape}")
#                 pos[:, group] -= offset.unsqueeze(1)
                pos = paddleindexjia(pos, -offset, group)

        # Move non-grouped atoms
        if len(self.nongrouped):
            offset = paddle.floor(pos[:, self.nongrouped] / box) * box
#             pos[:, self.nongrouped] -= offset.unsqueeze(1)
            pos = paddleindexjia(pos, -offset, self.nongrouped)


def calculate_molecule_groups(natoms, bonds):
    import networkx as nx
    import numpy as np

    # Calculate molecule groups and non-bonded / non-grouped atoms
    if bonds is not None and len(bonds):
        bondGraph = nx.Graph()
        bondGraph.add_nodes_from(range(natoms))
        bondGraph.add_edges_from(bonds.astype(np.int64))
        molgroups = list(nx.connected_components(bondGraph))

        nongrouped = paddle.to_tensor(
            [list(group)[0] for group in molgroups if len(group) == 1]
        ) 
        molgroups = [
            paddle.to_tensor(list(group)) 
            for group in molgroups
            if len(group) > 1
        ]
    else:
        molgroups = []
        nongrouped = paddle.arange(0, natoms) 
    return molgroups, nongrouped


# # minimizers.py
# 

# In[22]:


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
        with Benchmark("sforces.compute的耗时"):
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
    with Benchmark("scipy.minimize的耗时"):
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
        stop_gradient=system.pos.stop_gradient
    )

def minimize_pytorch_bfgs(system, forces, steps=1000):
    if steps == 0:
        return

    pos = system.pos.detach().requires_grad_(True)
    opt = paddle.optim.LBFGS([pos], max_iter=steps, tolerance_change=1e-09)

    def closure(step):
        opt.zero_grad()
        Epot = forces.compute(
            pos, system.box, system.forces, explicit_forces=False, returnDetails=False
        )
        Etot = paddle.sum(paddle.cat(Epot))
        grad = -system.forces.detach().numpy().astype(np.float64)[0]
        maxforce = float(paddle.max(paddle.norm(grad, axis=1)))
        print("{0:4d}   {1: 3.6f}   {2: 3.6f}".format(step[0], float(Etot), maxforce))
        step[0] += 1
        return Etot

    print("{0:4s} {1:9s}       {2:9s}".format("Iter", " Epot", " fmax"))
    step = [0]
    opt.step(lambda: closure(step))
#     with Benchmark("测试")
    system.pos[:] = pos.detach().requires_grad_(False)


# # 其它几个文件
# 
# utils.py 不用修改

# In[23]:


import csv
import json
import os
import time
import argparse
import yaml


class LogWriter(object):
    # kind of inspired form openai.baselines.bench.monitor
    # We can add here an optional Tensorboard logger as well
    def __init__(self, path, keys, header="", name="monitor.csv"):
        self.keys = tuple(keys) + ("t",)
        assert path is not None

        os.makedirs(path, exist_ok=True)
        filename = os.path.join(path, name)
        if os.path.exists(filename):
            os.remove(filename)

        print("Writing logs to ", filename)

        self.f = open(filename, "wt")
        if isinstance(header, dict):
            header = "# {} \n".format(json.dumps(header))
        self.f.write(header)
        self.logger = csv.DictWriter(self.f, fieldnames=self.keys)
        self.logger.writeheader()
        self.f.flush()
        self.tstart = time.time()

    def write_row(self, epinfo):
        if self.logger:
            t = time.time() - self.tstart
            epinfo["t"] = t
            self.logger.writerow(epinfo)
            self.f.flush()


class LoadFromFile(argparse.Action):
    # parser.add_argument('--file', type=open, action=LoadFromFile)
    def __call__(self, parser, namespace, values, option_string=None):
        if values.name.endswith("yaml") or values.name.endswith("yml"):
            with values as f:
                namespace.__dict__.update(yaml.load(f, Loader=yaml.FullLoader))
                return

        with values as f:
            input = f.read()
            input = input.rstrip()
            for lines in input.split("\n"):
                k, v = lines.split("=")
                typ = type(namespace.__dict__[k])
                v = typ(v) if typ is not None else v
                namespace.__dict__[k] = v


def save_argparse(args, filename, exclude=None):
    if filename.endswith("yaml") or filename.endswith("yml"):
        if isinstance(exclude, str):
            exclude = [
                exclude,
            ]
        args = args.__dict__.copy()
        for exl in exclude:
            del args[exl]
        with open(filename, "w") as fout:
            yaml.dump(args, fout)
    else:
        with open(filename, "w") as f:
            for k, v in args.__dict__.items():
                if k is exclude:
                    continue
                f.write(f"{k}={v}\n")


# # mycalc.py

# In[24]:


# !pip install ase  -i https://mirror.baidu.com/pypi/simple # 原子模拟环境Atomic Simulation Environment

# In[25]:


from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.calculator import InputError, ReadError
from ase.calculators.calculator import CalculatorSetupError
from ase import io
import numpy as np
from ase.units import Bohr, Hartree, kcal, mol, Angstrom
import os
import paddle


class MyCalc(Calculator):
    implemented_properties = ['energy', 'forces']

    def __init__(self, evaluator, restart=None, ignore_bad_restart=False,
                 label='mycalc', atoms=None, command=None,
                 **kwargs):
        Calculator.__init__(self, restart=restart,
                            ignore_bad_restart=ignore_bad_restart, label=label,
                            atoms=atoms, command=command, **kwargs)
        self.evaluator = evaluator

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=all_changes, symmetry='c1'):

        Calculator.calculate(self, atoms=atoms)
        if self.atoms is None:
            raise CalculatorSetupError('An Atoms object must be provided to '
                                       'perform a calculation')
        atoms = self.atoms

        pos = paddle.to_tensor(atoms.positions).double()
        cell = atoms.cell.tolist()
        cell = paddle.to_tensor([cell[0][0], cell[1][1], cell[2][2]]).double()
        energy = self.evaluator.compute(pos, cell)
        
        # Do the calculations
        if 'forces' in properties:
            # energy comes for free
            self.results['energy'] = energy
            # convert to eV/A
            # also note that the gradient is -1 * forces
            self.results['forces'] = self.evaluator.forces.numpy()
        elif 'energy' in properties:
            # convert to eV
            self.results['energy'] = energy




# # neighbourlist.py

# In[26]:


import paddle

# 写飞桨版本的笛卡尔直积函数cartesian_prod
from itertools import product
def paddlecartesian_prod(x,y):
    z = list(product(x,y))
    z = paddle.to_tensor(z)
    return z.squeeze(axis=-1)

def discretize_box(box, subcell_size):
    xbins = paddle.arange(0, box[0, 0] + subcell_size, subcell_size)
    ybins = paddle.arange(0, box[1, 1] + subcell_size, subcell_size)
    zbins = paddle.arange(0, box[2, 2] + subcell_size, subcell_size)
    nxbins = len(xbins) - 1
    nybins = len(ybins) - 1
    nzbins = len(zbins) - 1

    r = paddle.to_tensor([-1, 0, 1])
    neighbour_mask = paddlecartesian_prod(r, r, r)

    cellidx = paddlecartesian_prod(
        paddle.arange(nxbins), paddle.arange(nybins), paddle.arange(nzbins)
    )
    cellneighbours = cellidx.unsqueeze(2) + neighbour_mask.T.unsqueeze(0).repeat(
        cellidx.shape[0], 1, 1
    )

    # Can probably be done easier as we only need to handle -1 and max cases, not general -2, max+1 etc
    nbins = paddle.to_tensor([nxbins, nybins, nzbins])[None, :, None].repeat(
        cellidx.shape[0], 1, 27
    )
    negvals = cellneighbours < 0
    cellneighbours[negvals] += nbins[negvals]
    largevals = cellneighbours > (nbins - 1)
    cellneighbours[largevals] -= nbins[largevals]

    return xbins, ybins, zbins, cellneighbours


# def neighbour_list(pos, box, subcell_size):
#     nsystems = coordinates.shape[0]

#     for s in range(nsystems):
#         spos = pos[s]
#         sbox = box[s]

#         xbins, ybins, zbins = discretize_box(sbox, subcell_size)

#         xidx = paddle.bucketize(spos[:, 0], xbins, out_int32=True)
#         yidx = paddle.bucketize(spos[:, 1], ybins, out_int32=True)
#         zidx = paddle.bucketize(spos[:, 2], zbins, out_int32=True)

#         binidx = paddle.stack((xidx, yidx, zidx)).T


# In[27]:


from moleculekit.molecule import Molecule
import os

testdir = "./test-data/prod_alanine_dipeptide_amber/"
mol = Molecule(os.path.join(testdir, "structure.prmtop"))  # Reading the system topology
mol.read(os.path.join(testdir, "input.coor"))  # Reading the initial simulation coordinates
mol.read(os.path.join(testdir, "input.xsc"))  # Reading the box dimensions

# In[28]:


# from paddlemd.forcefields.forcefield import ForceField
# from paddlemd.parameters import Parameters
import paddle

precision = paddle.float32
# device = "cuda:0"

ff = ForceField.create(mol, os.path.join(testdir, "structure.prmtop"))
parameters = Parameters(ff, mol, precision=precision)

# In[29]:


# create system
# from paddlemd.integrator import maxwell_boltzmann
# from paddlemd.systems import System

system = System(mol.numAtoms, nreplicas=1, precision=precision)
system.set_positions(mol.coords)
system.set_box(mol.box)
system.set_velocities(maxwell_boltzmann(parameters.masses, T=300, replicas=1))

# In[30]:


from paddlemd.forces import Forces
bonded = ["bonds", "angles", "dihedrals", "impropers", "1-4"]
# bonded = ["dihedrals"]
# forces = Forces(parameters, cutoff=9, rfa=True, switch_dist=7.5)
forces = Forces(parameters, cutoff=9, rfa=True, switch_dist=7.5, terms=bonded)
# Evaluate current energy and forces. Forces are modified in-place
Epot = forces.compute(system.pos, system.box, system.forces, returnDetails=True)

print(Epot)
print(system.forces)

# In[31]:


from paddlemd.integrator import Integrator
from paddlemd.wrapper import Wrapper

langevin_temperature = 300  # K
langevin_gamma = 0.1
timestep = 1  # fs

integrator = Integrator(system, forces, timestep, gamma=langevin_gamma, T=langevin_temperature)
wrapper = Wrapper(mol.numAtoms, mol.bonds if len(mol.bonds) else None)

# In[32]:


# 开始训练
# from paddlemd.minimizers import minimize_bfgs

# 
minimize_bfgs(system, forces, steps=20)  # Minimize the system steps=500

# In[33]:


# 加日志？
# from paddlemd.utils import LogWriter

logger = LogWriter(path="logs/", keys=('iter','ns','epot','ekin','etot','T'), name='monitor.csv')

# In[34]:


# 终于可以跑了
from tqdm import tqdm 
import numpy as np

FS2NS = 1E-6 # Femtosecond to nanosecond conversion

steps = 10 # 1000
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
    print(i)
    
    if (i*output_period) % save_period  == 0:
        np.save(trajectoryout, np.stack(traj, axis=2))

    logger.write_row({'iter':i*output_period,'ns':FS2NS*i*output_period*timestep,'epot':Epot,'ekin':Ekin,'etot':Epot+Ekin,'T':T})

# # run.py
# 

# In[35]:


import os
import paddle
# from paddlemd.systems import System
from moleculekit.molecule import Molecule
# from paddlemd.forcefields.forcefield import ForceField
# from paddlemd.parameters import Parameters
# from paddlemd.forces import Forces
# from paddlemd.integrator import Integrator
# from paddlemd.wrapper import Wrapper
import numpy as np
from tqdm import tqdm
import argparse
import math
import importlib
# from paddlemd.integrator import maxwell_boltzmann
# from paddlemd.utils import save_argparse, LogWriter,LoadFromFile
# from paddlemd.minimizers import minimize_bfgs

FS2NS=1E-6


def viewFrame(mol, pos, forces):
    from ffevaluation.ffevaluate import viewForces
    mol.coords[:, :, 0] = pos[0].numpy()
    mol.view(guessBonds=False)
    viewForces(mol, forces[0].numpy()[:, :, None] * 0.01)

def get_args(arguments=None):
    parser = argparse.ArgumentParser(description='TorchMD',prefix_chars='--')
    parser.add_argument('--conf', type=open, action=LoadFromFile, help='Use a configuration file, e.g. python run.py --conf input.conf')
    parser.add_argument('--timestep', default=1, type=float, help='Timestep in fs')
    parser.add_argument('--temperature',  default=300,type=float, help='Assign velocity from initial temperature in K')
    parser.add_argument('--langevin-temperature',  default=0,type=float, help='Temperature in K of the thermostat')
    parser.add_argument('--langevin-gamma',  default=0.1,type=float, help='Langevin relaxation ps^-1')
    parser.add_argument('--device', default='cpu', help='Type of device, e.g. "cuda:1"')
    parser.add_argument('--structure', default=None, help='Deprecated: Input PDB')
    parser.add_argument('--topology', default=None, type=str, help='Input topology')
    parser.add_argument('--coordinates', default=None, type=str, help='Input coordinates')
    parser.add_argument('--forcefield', default="tests/argon/argon_forcefield.yaml", help='Forcefield .yaml file')
    parser.add_argument('--seed',type=int,default=1,help='random seed (default: 1)')
    parser.add_argument('--output-period',type=int,default=10,help='Store trajectory and print monitor.csv every period')
    parser.add_argument('--save-period',type=int,default=0,help='Dump trajectory to npy file. By default 10 times output-period.')
    parser.add_argument('--steps',type=int,default=10000,help='Total number of simulation steps')
    parser.add_argument('--log-dir', default='./', help='Log directory')
    parser.add_argument('--output', default='output', help='Output filename for trajectory')
    parser.add_argument('--forceterms', nargs='+', default="LJ", help='Forceterms to include, e.g. --forceterms Bonds LJ')
    parser.add_argument('--cutoff', default=None, type=float, help='LJ/Elec/Bond cutoff')
    parser.add_argument('--switch_dist', default=None, type=float, help='Switching distance for LJ')
    parser.add_argument('--precision', default='single', type=str, help='LJ/Elec/Bond cutoff')
    parser.add_argument('--external', default=None, type=dict, help='External calculator config')
    parser.add_argument('--rfa', default=False, action='store_true', help='Enable reaction field approximation')
    parser.add_argument('--replicas', type=int, default=1, help='Number of different replicas to run')
    parser.add_argument('--extended_system', default=None, type=float, help='xsc file for box size')
    parser.add_argument('--minimize', default=None, type=int, help='Minimize the system for `minimize` steps')
    parser.add_argument('--exclusions', default=('bonds', 'angles', '1-4'), type=tuple, help='exclusions for the LJ or repulsionCG term')
    
#     args = parser.parse_args(args=arguments)
    args = parser.parse_args(['--seed', '42', "--topology", "./test-data/prod_alanine_dipeptide_amber/structure.prmtop"])
#     print(args)
#     args = parser.parse_args(['--device', 'gpu'])
    os.makedirs(args.log_dir,exist_ok=True)
    save_argparse(args,os.path.join(args.log_dir,'input.yaml'),exclude='conf')

    if isinstance(args.forceterms, str):
        args.forceterms = [args.forceterms]
    if args.steps%args.output_period!=0:
        raise ValueError('Steps must be multiple of output-period.')
    if args.save_period == 0:
        args.save_period = 10*args.output_period
    if args.save_period%args.output_period!=0:
        raise ValueError('save-period must be multiple of output-period.')

    return args

precisionmap = {'single': paddle.float32, 'double': paddle.float64}

def setup(args):
    paddle.seed(args.seed)
#     paddle.cuda.manual_seed_all(args.seed)
    #We want to set TF32 to false by default to avoid precision problems
#     paddle.backends.cuda.matmul.allow_tf32 = False
#     paddle.backends.cudnn.allow_tf32 = False
#     device = paddle.device(args.device)

    if args.topology is not None:
        mol = Molecule(args.topology)
    elif args.structure is not None:
        mol = Molecule(args.structure)
        mol.box = np.array([mol.crystalinfo['a'],mol.crystalinfo['b'],mol.crystalinfo['c']]).reshape(3, 1).astype(np.float32)

    if args.coordinates is not None:
        mol.read(args.coordinates)

    if args.extended_system is not None:
        mol.read(args.extended_system)

    precision = precisionmap[args.precision]

    print("Force terms: ",args.forceterms)
    ff = ForceField.create(mol, args.forcefield)
#     parameters = Parameters(ff, mol, args.forceterms, precision=precision, device=device)
    parameters = Parameters(ff, mol, args.forceterms, precision=precision)

    external = None
    if args.external is not None:
        externalmodule = importlib.import_module(args.external["module"])
        embeddings = paddle.to_tensor(args.external["embeddings"]).repeat(args.replicas, 1)
        external = externalmodule.External(args.external["file"], embeddings)

    system = System(mol.numAtoms, args.replicas, precision)
    system.set_positions(mol.coords)
    system.set_box(mol.box)
    system.set_velocities(maxwell_boltzmann(parameters.masses, args.temperature, args.replicas))

    forces = Forces(parameters, terms=args.forceterms, external=external, cutoff=args.cutoff, rfa=args.rfa, switch_dist=args.switch_dist, exclusions=args.exclusions)
    return mol, system, forces

def dynamics(args, mol, system, forces):
    paddle.seed(args.seed)
#     paddle.cuda.manual_seed_all(args.seed)
#     device = paddle.device(args.device)

    integrator = Integrator(system, forces, args.timestep, gamma=args.langevin_gamma, T=args.langevin_temperature)
    wrapper = Wrapper(mol.numAtoms, mol.bonds if len(mol.bonds) else None)

    outputname, outputext = os.path.splitext(args.output)
    trajs = []
    logs = []
    for k in range(args.replicas):
        logs.append(LogWriter(args.log_dir,keys=('iter','ns','epot','ekin','etot','T'), name=f'monitor_{k}.csv'))
        trajs.append([])

    if args.minimize != None:
        minimize_bfgs(system, forces, steps=args.minimize)

    iterator = tqdm(range(1,int(args.steps/args.output_period)+1))
    Epot = forces.compute(system.pos, system.box, system.forces)
    for i in iterator:
        # viewFrame(mol, system.pos, system.forces)
        Ekin, Epot, T = integrator.step(niter=args.output_period)
        wrapper.wrap(system.pos, system.box)
        currpos = system.pos.detach().numpy().copy()
        for k in range(args.replicas):
            trajs[k].append(currpos[k])
            if (i*args.output_period) % args.save_period  == 0:
                np.save(os.path.join(args.log_dir, f"{outputname}_{k}{outputext}"), np.stack(trajs[k], axis=2)) #ideally we want to append
            
            logs[k].write_row({'iter':i*args.output_period,'ns':FS2NS*i*args.output_period*args.timestep,'epot':Epot[k],
                                'ekin':Ekin[k],'etot':Epot[k]+Ekin[k],'T':T[k]})
        
                

if __name__ == "__main__":
    args = get_args()
#     opt = args.parse_args(['--seed', '1'])
    mol, system, forces = setup(args)
    dynamics(args, mol, system, forces)




# # 代码复现第二阶段
# 
# 像常规notebook下的调试流程
# ##  1、对疑点文件拆分，将函数放到Cell进行测试
# 
# 测试中可以加入测试代码，验证函数是否正确。最终保证所有函数测试通过
# ##  2、测试通过后，将修改写回文件
# ## 3、在tutorial.ipynb文件中总测试
# 
# 优点是，基本不修改tutorial.ipynb文件代码。

# In[ ]:


# !pwd
# !python run.py
# 报错 ModuleNotFoundError: No module named 'paddlemd' ，把run.py放到上一个目录就行了。

# In[ ]:


2

# # 代码复现第三阶段
# 调试精度和速度
# 
# ## 1、速度提高
# 有些手工写的算子使用了for循环，需要找到并提速。
# 
# ## 2、精度测试

# In[ ]:




# # 调试
# ## 安装moleculekit失败
# 报错信息在：/home/aistudio/.webide/3863645/log/install-moleculekit-2022-04-22-22-13-08.log
# ```
# has inconsistent version: filename has '1.0.0', but metadata has '0'
# ERROR: Could not find a version that satisfies the requirement moleculekit==1.0.0 (from versions: 0.1.4, 0.1.5, 0.1.6, 0.1.7, 0.1.8, 0.1.9, 0.1.12, 0.1.14, 0.1.15, 0.1.16, 0.1.17, 0.1.19, 0.1.21, 0.1.22, 0.1.23, 0.1.24, 0.1.26, 0.1.27, 0.1.29, 0.1.30, 0.1.31, 0.1.32, 0.2.0, 0.2.1, 0.2.2, 0.2.3, 0.2.4, 0.2.5, 0.2.6, 0.2.7, 0.2.8, 0.2.9, 0.3.0, 0.3.1, 0.3.2, 0.3.3, 0.3.4, 0.3.5, 0.3.7, 0.3.8, 0.3.9, 0.4.0, 0.4.2, 0.4.3, 0.4.4, 0.4.6, 0.4.7, 0.4.8, 0.5.2, 0.5.3, 0.5.4, 0.5.5, 0.5.6, 0.5.7, 0.5.8, 0.5.9, 0.6.0, 0.6.1, 0.6.3, 0.6.4, 0.6.5, 0.6.7, 0.6.8, 0.7.0, 0.7.1, 0.7.2, 0.7.3, 0.7.4, 0.7.5, 0.7.6, 0.7.7, 0.7.8, 0.7.9, 0.8.0, 0.8.1, 0.8.2, 0.8.3, 0.8.5, 0.8.6, 0.8.9, 0.9.0, 0.9.1, 0.9.2, 0.9.3, 0.9.4, 0.9.5, 0.9.6, 0.9.7, 0.9.8, 0.9.9, 0.9.12, 0.9.13, 0.9.14, 0.9.15, 0.10.0, 1.0.0, 1.0.1, 1.0.2, 1.0.3, 1.0.4, 1.0.5, 1.0.6, 1.0.7, 1.0.8, 1.0.9, 1.1.0, 1.1.1, 1.1.2, 1.1.4, 1.1.6, 1.1.7, 1.1.8, 1.1.9, 1.2.0)
# ERROR: No matching distribution found for moleculekit==1.0.0
# ```
# 通过源码编译安装，终于装上了
# ```
# # !wget https://files.pythonhosted.org/packages/06/90/69685dad023515e231d8da7747793f33c497bb62496c4075e518f230da55/moleculekit-1.2.0.tar.gz
# # !tar -xzvf moleculekit-1.2.0.tar.gz
# # !cd moleculekit-1.2.0/ && python setup.py install 
# ```
# ## TypeError: dtype must be a type, str, or dtype object
# ```
# /code/6paper/PaddleMD/paddlemd/parameters.py in precision_(self, precision)
#      60 
#      61     def precision_(self, precision):
# ---> 62         self.A = self.A.type(precision)
#      63         self.B = self.B.type(precision)
#      64         self.charges = self.charges.type(precision)
# 
# TypeError: dtype must be a type, str, or dtype object
# ```
# 反正改成paddle.float32
# 
# ## 报错'paddle.fluid.core_avx.VarType' object is not callable
# ```
# ---> 63         self.B = self.B.type(precision)
#      64         self.charges = self.charges.type(precision)
#      65         self.masses = self.masses.type(precision)
# 
# TypeError: 'paddle.fluid.core_avx.VarType' object is not callable
# ```
# 修改self.box.type(precision)
# 改成astype即可。
# 
# ## name 'device' is not defined
# ```
# ---> 32         self.to_(device)
#      33 
#      34     def to_(self, device):
# 
# NameError: name 'device' is not defined
# ```
# torch里面的device，应该是可以全去掉的。
# 首先查找替换，把.to(device)全部替换为没有（实际上用了一个空格）
# 
# 然后修改self.to_(device)这句，全局有一处：
# systems.py中有有一处，删除.to_函数和相应调用语句
# 
# 除run.py文件里面没有修改之外，将文件中的device去掉或注释掉。
# 
# ## 报错'Tensor' object has no attribute 'bool'
# ```python
# ----> 6 system.set_box(mol.box)
#       7 system.set_velocities(maxwell_boltzmann(parameters.masses, T=300, replicas=1))
# 
# /code/6paper/PaddleMD/paddlemd/systems.py in set_box(self, box)
#      71         for r in range(box.shape[0]):
#      72             self.box[r][paddle.eye(3).bool()] = paddle.to_tensor(
# ---> 73                 box[r], dtype=self.box.dtype)
#      74 
#      75     def set_forces(self, forces):
# 
# AttributeError: 'Tensor' object has no attribute 'bool'
# ```
# 将bool()改成astype(paddle.bool)
# 
# # 报错TypeError: data type not understood
# ```
# ----> 7 system.set_velocities(maxwell_boltzmann(parameters.masses, T=300, replicas=1))
# 
# /code/6paper/PaddleMD/paddlemd/integrator.py in maxwell_boltzmann(masses, T, replicas)
#      16     for i in range(replicas):
#      17         velocities.append(
# ---> 18             paddle.sqrt(T * BOLTZMAN / masses) * paddle.randn((natoms, 3)).astype(masses)
#      19         )
#      20 
# --> 731     dtype = np.dtype(np_dtype)
#     732     if dtype == np.float32:
#     733         return core.VarDesc.VarType.FP32
# 
# TypeError: data type not understood
# ```
# 将astype(masses)改成.astype(masses.dtype)
# # stack() got an unexpected keyword argument 'dim'
# ```
# ----> 7 system.set_velocities(maxwell_boltzmann(parameters.masses, T=300, replicas=1))
# 
# /code/6paper/PaddleMD/paddlemd/integrator.py in maxwell_boltzmann(masses, T, replicas)
#      19         )
#      20 
# ---> 21     return paddle.stack(velocities, axis=0)
#      22 
#      23 
# 
# TypeError: stack() got an unexpected keyword argument 'dim'
# ```
# 将全部的dim= 改成axis=
# 
# # 报错 Velocities shape must be (nreplicas, natoms, 3)
# ```
# /code/6paper/PaddleMD/paddlemd/systems.py in set_velocities(self, vel)
#      52     def set_velocities(self, vel):
#      53         if vel.shape != (self.nreplicas, self.natoms, 3):
# ---> 54             raise RuntimeError("Velocities shape must be (nreplicas, natoms, 3)")
#      55         self.vel[:] = vel.clone().detach().astype(self.vel.dtype)
#      56 
# 
# RuntimeError: Velocities shape must be (nreplicas, natoms, 3)
# ```
# 飞桨里面shape返回的是列表，所以这里要修改下.修改了2处
# 
# # 粗心，出来一个中文括号
# ```
#   File "/code/6paper/PaddleMD/paddlemd/forces.py", line 547
#     coeff = paddle.zeros(ntorsions, dtype=r12.dtype）
#                                                    ^
# SyntaxError: invalid character in identifier
# ```
# 
# # Expected size == 1, but received size:2 != 1:1.] 
# ```python
# ----> 7 Epot = forces.compute(system.pos, system.box, system.forces, returnDetails=True)
#       8 
#       9 print(Epot)
# 
# /code/6paper/PaddleMD/paddlemd/forces.py in compute(self, pos, box, forces, returnDetails, explicit_forces)
#     137             if "dihedrals" in self.energies and self.par.dihedrals is not None:
#     138                 _, _, r12 = calculate_distances(
# --> 139                     spos, self.par.dihedrals[:, [0, 1]], sbox
#     140                 )
#     141                 _, _, r23 = calculate_distances(
# 
# /opt/conda/lib/python3.6/site-packages/paddle/fluid/dygraph/varbase_patch_methods.py in __getitem__(self, item)
#     596         else:
#     597             # 2. Call c++ func getitem_index_not_tensor to speedup.
# --> 598             return self._getitem_index_not_tensor(item)
#     599 
#     600     def __setitem__(self, item, value):
# 
# ValueError: (InvalidArgument) When index contains a list, its length is excepted to 1, but received 2
#   [Hint: Expected size == 1, but received size:2 != 1:1.] (at /paddle/paddle/fluid/pybind/imperative.cc:599)
# ```
# 查找原因，是torch和飞桨的切片不一样，torch这样切：self.par.dihedrals[:, [0, 1]]
# 而飞桨应该self.par.dihedrals[:, 0:2]]。 果然，修改之后就好了。
# 顺便把所有的切片都改好.
# 需要注意的是反序切片：[:, [2, 1]] 需要修改成 [:, 2:0:-1]
# 
# ## paddle.zeros(ntorsions, dtype=r12.dtype) 报错
# ```python
# ----> 7 Epot = forces.compute(system.pos, system.box, system.forces, returnDetails=True)
#       8 
#       9 print(Epot)
# 
# /code/6paper/PaddleMD/paddlemd/forces.py in compute(self, pos, box, forces, returnDetails, explicit_forces)
#     155                 )
#     156                 E, dihedral_forces = evaluate_torsion(
# --> 157                     r12, r23, r34, self.par.dihedral_params, explicit_forces
#     158                 )
#     159 
# 
# /code/6paper/PaddleMD/paddlemd/forces.py in evaluate_torsion(r12, r23, r34, torsion_params, explicit_forces)
#     551     ntorsions = len(torsion_params[0]["idx"])
#     552 #     pot = paddle.zeros(ntorsions, dtype=r12.dtype, layout=r12.layout)
# --> 553     pot = paddle.zeros(ntorsions, dtype=r12.dtype) # 飞桨无layout参数
#     554     if explicit_forces:
#     555         coeff = paddle.zeros(
# --> 369         shape = shape.numpy().astype(int).tolist()
#     370     return shape
#     371 
# 
# AttributeError: 'int' object has no attribute 'numpy'
# ```
# 应该是zeros的shape参数需要用列表的缘故。全部加上中括号paddle.zeros([ntorsions], dtype=...
# 
# ## 'Tensor' object has no attribute 'layout'
# ```python
#     554     if explicit_forces:
#     555         coeff = paddle.zeros(
# --> 556             [ntorsions], dtype=r12.dtype, layout=r12.layout)
#     557         coeff = paddle.zeros([ntorsions], dtype=r12.dtype)
#     558     for i in range(0, len(torsion_params)):
# 
# AttributeError: 'Tensor' object has no attribute 'layout'
# ```
# 找到所有的layout，去掉 
# 
# ##  'Tensor' object has no attribute 'scatter_add_'
# ```python
# /code/6paper/PaddleMD/paddlemd/forces.py in evaluate_torsion(r12, r23, r34, torsion_params, explicit_forces)
#     564         if paddle.all(per > 0):  # AMBER torsions
#     565             angleDiff = per * phi[idx] - phi0
# --> 566             pot.scatter_add_(0, idx, k0 * (1 + paddle.cos(angleDiff)))
#     567             if explicit_forces:
#     568                 coeff.scatter_add_(0, idx, -per * k0 * paddle.sin(angleDiff))
# 
# AttributeError: 'Tensor' object has no attribute 'scatter_add_'
# ```
# 用自己写的paddlescatter代替看看。
# 
# ## 自己的scatter报错
# ```python
# /code/6paper/PaddleMD/paddlemd/forces.py in paddlescatter(x, dim, index, src)
#      14 def paddlescatter(x, dim, index, src):
#      15     updates = src
# ---> 16     i, j = index.shape
#      17     grid_x , grid_y = paddle.meshgrid(paddle.arange(i), paddle.arange(j))
#      18     if dim == 0 :
# 
# ValueError: not enough values to unpack (expected 2, got 1)
# ```
# 难道index是1D的？ 跟踪输出，发现果然是1D数据，长度是41 。
# 
# 这样就需要针对1D数据写paddlescatter，因为以前的代码是针对2D数据的！
# 重新写了支持1D的版本
# ```python
# def paddlescatter(x, dim, index, src): # 支持1D版本
#     
#     updates = src
#     if len(index.shape) == 1 :
#         for i in index:
#             x[i] += updates[i]
#         return x
#                                 
#     i, j = index.shape
#     grid_x , grid_y = paddle.meshgrid(paddle.arange(i), paddle.arange(j))
#     if dim == 0 :
#         index = paddle.stack([index.flatten(), grid_y.flatten()], axis=1)
#     elif dim == 1:
#         index = paddle.stack([grid_x.flatten(), index.flatten()], axis=1)
#     # 若 PyTorch 的 dim 取 0
#     # index = paddle.stack([index.flatten(), grid_y.flatten()], axis=1)
#     # 若 PyTorch 的 dim 取 1
#     # index = paddle.stack([grid_x.flatten(), index.flatten()], axis=1)
#     # PaddlePaddle updates 的 shape 大小必须与 index 对应
#     updates_index = paddle.stack([grid_x.flatten(), grid_y.flatten()], axis=1)
#     updates = paddle.gather_nd(updates, index=updates_index)
#     return paddle.scatter_nd_add(x, index, updates)
# ```
# 修改完毕后，有新的报错：
# ## paddlescatter还是有问题
# ```python
# /code/6paper/PaddleMD/paddlemd/forces.py in paddlescatter(x, dim, index, src)
#      17     if len(index.shape) == 1 :
#      18         for i in index:
# ---> 19             x[i] += updates[i]
#      20         return x
#      21 
# 
# /opt/conda/lib/python3.6/site-packages/paddle/fluid/dygraph/varbase_patch_methods.py in __getitem__(self, item)
#     592             # 1. Call _getitem_impl_ when item contains tensor.
#     593             # Why not call a c++ function ? Because item can't be parsed when it contains tensor.
# --> 594             return _getitem_impl_(self, item)
#     595 
#     596         else:
# 
# /opt/conda/lib/python3.6/site-packages/paddle/fluid/variable_index.py in _getitem_impl_(var, item)
#     462             inputs=inputs,
#     463             outputs={'Out': [slice_out_var]},
# --> 464             attrs=attrs)
#     465         out = slice_out_var
#     466 
# 
# /opt/conda/lib/python3.6/site-packages/paddle/fluid/framework.py in append_op(self, *args, **kwargs)
#    3165                                        kwargs.get("outputs", {}), attrs
#    3166                                        if attrs else {},
# -> 3167                                        kwargs.get("stop_gradient", False))
#    3168         else:
#    3169             from paddle.fluid.dygraph.base import param_guard
# 
# /opt/conda/lib/python3.6/site-packages/paddle/fluid/dygraph/tracer.py in trace_op(self, type, inputs, outputs, attrs, stop_gradient)
#      43         self.trace(type, inputs, outputs, attrs,
#      44                    framework._current_expected_place(), self._has_grad and
# ---> 45                    not stop_gradient)
#      46 
#      47     def train_mode(self):
# 
# ValueError: (InvalidArgument) When step > 0, end should be greater than start, but received end = 10, start = 10.
#   [Hint: Expected end > start, but received end:10 <= start:10.] (at /paddle/paddle/fluid/operators/slice_utils.h:59)
#   [operator < slice > error]
# 
# ```
# 不明白为什么这里会报错。不过再回头测试自己写的paddlescatter函数，发现在x和src长度不一样的情况下，跟torch没有对齐 。
# 好了，终于对齐了，原来前面有逻辑问题：
# ```python
# def paddlescatter(x, dim, index, src): # 支持1D版本
#     
#     updates = src
#     if len(index.shape) == 1 :
# #         for i in index:
# #             x[i] += updates[i]
#         for i in range(index.shape[0]):
#             x[index[i]] += updates[i]
#         return x
# ```
# 调试通过，新的报错：
# ## 'Tensor' object has no attribute 'index_add_'
# 飞桨没有index_add_啊！
# 写了一个
# ```python
# def paddleindex_add(x, dim, index, source): # 飞桨的index_add
#     for i in range(len(index)):
#         x[index[i]] += source[i]
#     return x
# ```
# 全部替换后测试通过，新的报错
# ##
# ```python
# /code/6paper/PaddleMD/paddlemd/minimizers.py in evalfunc(coords, info)
#      16     def evalfunc(coords, info):
#      17         coords = coords.reshape(1, -1, 3)
# ---> 18         coords = paddle.to_tensor(coords).astype(system.pos)
#      19         Epot = forces.compute(coords, system.box, system.forces)[0]
#      20         grad = -system.forces.detach().cpu().numpy().astype(np.float64)[0]
# 
# /opt/conda/lib/python3.6/site-packages/paddle/fluid/dygraph/math_op_patch.py in astype(self, dtype)
#     103         """
#     104         if not isinstance(dtype, core.VarDesc.VarType):
# --> 105             dtype = convert_np_dtype_to_dtype_(dtype)
#     106         return _C_ops.cast(self, 'in_dtype', self.dtype, 'out_dtype', dtype)
#     107 
# 
# /opt/conda/lib/python3.6/site-packages/paddle/fluid/framework.py in convert_np_dtype_to_dtype_(np_dtype)
#     729 
#     730     """
# --> 731     dtype = np.dtype(np_dtype)
#     732     if dtype == np.float32:
#     733         return core.VarDesc.VarType.FP32
# 
# TypeError: data type not understood
# ```
# 修改成`coords = paddle.to_tensor(coords).astype(system.pos.dtype)`
# 测试通过，在训练298步之后，新的报错
# ## 'Tensor' object has no attribute 'requires_grad'
# ```python
# ----> 3 minimize_bfgs(system, forces, steps=500)  # Minimize the system
# 
# /code/6paper/PaddleMD/paddlemd/minimizers.py in minimize_bfgs(system, forces, fmax, steps)
#      44         res.x.reshape(1, -1, 3),
#      45         dtype=system.pos.dtype,
# ---> 46         requires_grad=system.pos.requires_grad,
#      47     )
#      48 
# 
# AttributeError: 'Tensor' object has no attribute 'requires_grad'
# ```
# 同时观察到训练速度慢，因为我的好几个函数都是for循环的。
# 改写成
# `stop_gradient=system.pos.stop_gradient`
# 测试通过。现在运行到最后一个cell，报错
# ## Valid index accept
# ```python
# /code/6paper/PaddleMD/paddlemd/wrapper.py in wrap(self, pos, box, wrapidx)
#       8     def wrap(self, pos, box, wrapidx=None):
#       9         nmol = len(self.groups)
# ---> 10         box = box[:, paddle.eye(3).astype(paddle.bool)]  # Use only the diagonal
#      11         if paddle.all(box == 0):
#      12             return
# 
# /opt/conda/lib/python3.6/site-packages/paddle/fluid/dygraph/varbase_patch_methods.py in __getitem__(self, item)
#     592             # 1. Call _getitem_impl_ when item contains tensor.
#     593             # Why not call a c++ function ? Because item can't be parsed when it contains tensor.
# --> 594             return _getitem_impl_(self, item)
#     595 
#     596         else:
# 
# /opt/conda/lib/python3.6/site-packages/paddle/fluid/variable_index.py in _getitem_impl_(var, item)
#     429             raise IndexError(
#     430                 "Valid index accept int or slice or ellipsis or list, but received {}.".
# --> 431                 format(item))
#     432         return slice_info.get_item(var)
#     433 
# 
# IndexError: Valid index accept int or slice or ellipsis or list, but received [slice(None, None, None), Tensor(shape=[3, 3], dtype=bool, place=CUDAPlace(0), stop_gradient=True,
#        [[True , False, False],
#         [False, True , False],
#         [False, False, True ]])].
# ```
# 将代码修改为
# ```python
# #         box = box[:, paddle.eye(3).astype(paddle.bool)]  # Use only the diagonal
#         box = box[:][paddle.eye(3).astype(paddle.bool)]
# ```
# 出现新的报错
# ```
# IndexError: The dimension of bool index doesn't match indexed array along dimension 0, the target dimension is 1, but received 3.
# ```
# 需要好好看看eye函数，以及tensor的切片用法了 。
# 实在不行，再去跟踪数据，看数据是否对头。
# eye处理的box数据：`box.sahpe [1, 3, 3]` 
# 最终修改成这样
# ```python
#         box = box.reshape([3, 3]) # 先试试这样的shape可以不？ 
#         box = box* (paddle.eye(3).astype(paddle.bool))
#         print(f"== after eye box.sahpe {box.shape}")
#         box = box.reshape([-1, 3, 3])
# ```
# 测试通过！最终运行速度22分钟！
# 再次修改上面的指令，修改成`box[0] = box[0] * (paddle.eye(3).astype(paddle.bool)) # 速度15 torch速度9 `
# 
# ## 报错module 'paddle' has no attribute 'clamp'
# ```python
# /code/6paper/PaddleMD/paddlemd/forces.py in evaluate_angles(r21, r23, angle_params, explicit_forces)
#     536 
#     537     cos_theta = dotprod * norm21inv * norm23inv
# --> 538     cos_theta = paddle.clamp(cos_theta, -1, 1)
#     539     theta = paddle.acos(cos_theta)
#     540 
# 
# AttributeError: module 'paddle' has no attribute 'clamp'
# ```
# 好像是用slip解决`cos_theta = paddle.clip(cos_theta, -1, 1)`
# 
# ## 解决tensor二阶赋值问题
# ```python
# #             self.box[r][paddle.eye(3).astype(paddle.bool)] = paddle.to_tensor(
# #                 box[r], dtype=self.box.dtype)
#             self.box[r] = paddle.to_tensor(
#                 box[r], dtype=self.box.dtype) * paddle.eye(3).astype(paddle.bool)
# ```
# 修改之后出现新的报错：
# ```python
# /code/6paper/PaddleMD/paddlemd/wrapper.py in wrap(self, pos, box, wrapidx)
#      29             # Work out the COMs and offsets of every group and move group to [0, box] range
#      30             for i, group in enumerate(self.groups):
# ---> 31                 tmp_com = paddle.sum(pos[:, group], axis=1) / len(group)
#      32                 offset = paddle.floor(tmp_com / box) * box
#      33                 pos[:, group] -= offset.unsqueeze(1)
# 
# /opt/conda/lib/python3.6/site-packages/paddle/fluid/dygraph/varbase_patch_methods.py in __getitem__(self, item)
#     592             # 1. Call _getitem_impl_ when item contains tensor.
#     593             # Why not call a c++ function ? Because item can't be parsed when it contains tensor.
# --> 594             return _getitem_impl_(self, item)
#     595 
#     596         else:
# 
# /opt/conda/lib/python3.6/site-packages/paddle/fluid/variable_index.py in _getitem_impl_(var, item)
#     429             raise IndexError(
#     430                 "Valid index accept int or slice or ellipsis or list, but received {}.".
# --> 431                 format(item))
#     432         return slice_info.get_item(var)
#     433 
# 
# IndexError: Valid index accept int or slice or ellipsis or list, but received [slice(None, None, None), Tensor(shape=[22], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
#        [0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10, 11, 12, 13, 14, 15, 16, 17,
#         18, 19, 20, 21])].
# ```
# 看[issue](https://github.com/PaddlePaddle/Paddle/issues/37733)里面的描述:目前PaddlePaddle中可以使用bool索引，但是暂时不支持bool和其他类型的索引同时使用，像data[2, index > 0]这样普通integer索引和bool索引同时使用会报错。这个问题我们正在解决中，但正式支持预计还需要一段时间。
# 
# 这里很费了一段时间，找不到思路。后来想明白了，反正是格式一致，都是在第二列进行索引操作，直接用一句简单的gather就行了
# ```python
# #             com = paddle.sum(pos[:, wrapidx], axis=1) / len(wrapidx)
#             com = paddle.sum(paddle.gather(pos, wrapidx, axis=1), axis=1) / len(wrapidx)
# 
# ```
# 谢天谢地这两个地方过去了，但是有新的报错
# ## 报错
# ```python
# ---> 35                 pos[:, group] -= offset.unsqueeze(1)
#      36 
#      37         # Move non-grouped atoms
# 
# /opt/conda/lib/python3.6/site-packages/paddle/fluid/dygraph/varbase_patch_methods.py in __getitem__(self, item)
#     592             # 1. Call _getitem_impl_ when item contains tensor.
#     593             # Why not call a c++ function ? Because item can't be parsed when it contains tensor.
# --> 594             return _getitem_impl_(self, item)
#     595 
#     596         else:
# 
# /opt/conda/lib/python3.6/site-packages/paddle/fluid/variable_index.py in _getitem_impl_(var, item)
#     429             raise IndexError(
#     430                 "Valid index accept int or slice or ellipsis or list, but received {}.".
# --> 431                 format(item))
#     432         return slice_info.get_item(var)
#     433 
# 
# IndexError: Valid index accept int or slice or ellipsis or list, but received [slice(None, None, None), Tensor(shape=[22], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
#        [0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10, 11, 12, 13, 14, 15, 16, 17,
#         18, 19, 20, 21])].
# ```
# 也就是这样的赋值操作该怎么破？ ` pos[:, group] -= offset.unsqueeze(1)`
# 自己手工写了一个函数，用来处理上面那句话，但是之后之后报错，根据跟踪信息，是box的维度有问题。
# torch那边的维度是[1,3], 而飞桨这边的维度是[1,3,3]
# 需要查找前面代码的问题了。
# ```python
# def paddleindexjia (x, y, xindex):
#     '''
#     切片+索引，使用循环来解决切片问题，然后使用中间变量，来实现按照索引赋值
#     支持类似的语句pos[:, group] -= offset.unsqueeze(1)
#     '''
#     xlen = len(x)
#     assert len(x.shape) == 3 , "维度不一致,必须为3D数据"
# #     if len(y.shape) == 3 and y.shape[0] ==1 :
# #         y = paddle.squeeze(y)
#     assert len(y.shape) ==2 , "维度不一致，必须为2D数据"
#     for i in range(xlen):
#         tmp = x[i]
#         tmp[xindex] += y
#         x[i] = tmp
#     return x
# ```
# 修改eye部分 
# ```python
# def paddleeye(x, n):
#     tmp =x[0][paddle.eye(n).astype(paddle.bool)]
#     return tmp.unsqueeze_(0)
# 
# #         box[0] = box[0] * (paddle.eye(3).astype(paddle.bool)) # 速度15 torch速度9 
#         box = paddleeye(box, 3)
# ```
# 现在程序终于能跑通了。
# 
# 现在的问题是：
# 1 程序有些地方使用了x[0]来代替torch的x[:] ，存在隐患
# 2 速度很慢，需要提高速度。
# 
# 个人感觉有如下两个地方是弱项：
# 1 飞桨的切片索引操作，不能同时使用，在这个项目里感觉很不好用，
# 2 个人对整个项目还没有较深入的了解，所以只能机械的修改torch命令，其实很多地方，应该有很大的提速空间。
# 
# 

# # 第二阶段调试
# ## 运行run报错 No module named 'paddlemd
# ```python
#   File "paddlemd/run.py", line 3, in <module>
#     from paddlemd.systems import System
# ModuleNotFoundError: No module named 'paddlemd'
# ```
# 不明白为什么啊，为什么无法导入啊。
# 把run.py放到上一个目录就行了。
# 
# ## force.compter 慢
# ```python
#         for i in range(nsystems):
#             spos = pos[i]
#             sbox = box[i][paddle.eye(3).astype(paddle.bool)]  # Use only the diagonal
# 
# ```
# 
# ## 查找涉及cpu计算的部分，全部去掉
# 比如to("cpu")去掉。
# .cpu().numpy() 替换成.numpy()
# .cpu().detach() 去掉。
# v.cpu().item() 替换成 v.item()
# 
# ## 将run.py文件修改为可以在notebook下执行
# 只需要找到parser.parse_args处，并修改成类似
# ```python
# #     args = parser.parse_args(args=arguments)
#     args = parser.parse_args(['--seed', '42', "--topology", "./test-data/prod_alanine_dipeptide_amber/structure.prmtop"])
# ```
# 之所以加上"--topology"参数，是因为若没有相关参数的话，会报 mol 没有初始化的问题。 命令行参数，没有找到手册。
# 运行之后报错：
# ```python
# <ipython-input-8-010ed07e7714> in make_masses(self, ff, atomtypes)
#     146 
#     147     def make_masses(self, ff, atomtypes):
# --> 148         masses = paddle.to_tensor([ff.get_mass(at) for at in atomtypes])
#     149         masses.unsqueeze_(1)  # natoms,1
#     150         return masses
# 
# <ipython-input-8-010ed07e7714> in <listcomp>(.0)
#     146 
#     147     def make_masses(self, ff, atomtypes):
# --> 148         masses = paddle.to_tensor([ff.get_mass(at) for at in atomtypes])
#     149         masses.unsqueeze_(1)  # natoms,1
#     150         return masses
# 
# /code/6paper/PaddleMD/paddlemd/forcefields/ff_yaml.py in get_mass(self, at)
#      58 
#      59     def get_mass(self, at):
# ---> 60         return self.prm["masses"][at]
#      61 
#      62     def get_LJ(self, at):
# 
# KeyError: 'HC'
# ```
# 先不管它了，先调试速度慢的问题。
# 
# 
# ## 加入时间统计函数Timer和Benchmark
# class Timer: 
# ```python
# class Benchmark:
#     """用于测量运行时间"""
#     def __init__(self, description='Done'):
#         self.description = description
# 
#     def __enter__(self):
#         self.timer = Timer()
#         return self
# 
#     def __exit__(self, *args):
#         print(f'{self.description}: {self.timer.stop():.4f} sec')
# ```
# ## 发现速度慢
# torchmd sforces.compute的耗时: 0.0075 sec
# 飞桨的sforces.compute的耗时: 1.4716 sec
# 大约差了200倍 
# 
# ## 修改二阶赋值
# ```python
# #                     pot[i]["electrostatics"] += E.sum()
#                     tmp = pot[i]
#                     tmp["electrostatics"] += E.sum()
#                     pot[i] = tmp
# ```
# 
# ## 顺手修改eye相关函数
# ```python
# #             sbox = box[i][paddle.eye(3).astype(paddle.bool)]  # Use only the diagonal
#             sbox = paddleeye(box, 3) # 将tensor eye转为自定义函数
# ```

# In[ ]:


1

# In[ ]:



