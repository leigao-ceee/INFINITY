import paddle
from math import sqrt
import numpy as np


class Parameters:
    def __init__(
        self,
        ff,
        mol,
        terms=None,
        precision=paddle.float32,
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

    #         self.to_(device=None) # 为了调试步幅小，临时用None默认值代替。

    #     def to_(self, device=None):
    #         self.A = self.A
    #         self.B = self.B
    #         self.charges = self.charges
    #         self.masses = self.masses
    #         if self.bonds is not None:
    #             self.bonds = self.bonds
    #             self.bond_params = self.bond_params
    #         if self.angles is not None:
    #             self.angles = self.angles
    #             self.angle_params = self.angle_params
    #         if self.dihedrals is not None:
    #             self.dihedrals = self.dihedrals
    #             for j in range(len(self.dihedral_params)):
    #                 termparams = self.dihedral_params[j]
    #                 termparams["idx"] = termparams["idx"]
    #                 termparams["params"] = termparams["params"]
    #         if self.idx14 is not None:
    #             self.idx14 = self.idx14
    #             self.nonbonded_14_params = self.nonbonded_14_params
    #         if self.impropers is not None:
    #             self.impropers = self.impropers
    #             termparams = self.improper_params[0]
    #             termparams["idx"] = termparams["idx"]
    #             termparams["params"] = termparams["params"]
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
            s6 = sig**6
            s12 = s6 * s6
            A = eps * 4 * s12
            B = eps * 4 * s6
            nonbonded_14_params.append([A, B, scnb, scee])
        return paddle.to_tensor(nonbonded_14_params)


def calculate_AB(sigma, epsilon):
    # Lorentz - Berthelot combination rule
    sigma_table = 0.5 * (sigma + sigma[:, None])
    eps_table = np.sqrt(epsilon * epsilon[:, None])
    sigma_table_6 = sigma_table**6
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
