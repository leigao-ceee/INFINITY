from scipy import constants as const
import paddle
import numpy as np
from math import pi

import paddle
import math
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
        for i in range(len(index)):
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

# 飞桨的put_alone_axis支持shape不一致的情况，即indices和value比arr长或者短的情况。
# 需要做的，就是要把paddlemd里面的[687]改成[688,1] forcovec[687,3]改成[688, 3]
def paddleput_alone_axis(arr, indices, value, axis, reduce="add"):
#     print(f"==arr.shape:{arr.shape} indices.shape:{indices.shape} value.shape:{value.shape}")
    lenarr = arr.shape[0]
    lenindices = indices.shape[0]
    while lenarr < lenindices:
        arr = paddle.put_along_axis(arr, indices[:lenarr].reshape([-1, 1]), \
            value[:lenarr], axis, reduce=reduce)
        indices = indices[lenarr:]
        value = value[lenarr:]
        lenarr = arr.shape[0]
        lenindices = indices.shape[0]
    xs = lenarr - lenindices
    if xs >= 1:
        newindices = paddle.concat([indices, paddle.zeros([xs], dtype=paddle.int64)]).reshape([-1, 1])
        newvalue = paddle.concat([value, paddle.zeros([xs, value.shape[-1]])])
    else:
        newindices = indices.reshape([-1, 1])
        newvalue = value
    out = paddle.put_along_axis(arr, newindices, newvalue, axis, reduce=reduce)
    return out

# 为了跟程序里的参数序列对齐，尽量不修改代码，写paddleindex_add
def paddleindex_add(x, dim, index, source):
    return paddleput_alone_axis(x, index, source, dim)

# def paddleindex_add(x, dim, index, source): # 飞桨的index_add
#     return x
#     for i in range(len(index)):
#         x[index[i]] += source[i]
#     return x

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
        if terms is None: # 为了不报错，我也是拼了
            terms = self.terms
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
#         if not explicit_forces and not pos.requires_grad:
        if not explicit_forces and  pos.stop_gradient:
            raise RuntimeError(
                "The positions passed don't require gradients. Please use pos.stop_gradient=False pos.detach().requires_grad_(True) before passing."
            )

        nsystems = pos.shape[0]
        if paddle.any(paddle.isnan(pos)):
            raise RuntimeError("Found NaN coordinates.")

        pot = []
        for i in range(nsystems):
            pp = {
                v: paddle.zeros([1]).astype(pos.dtype)
                for v in self.energies
            }
            pp["external"] = paddle.zeros([1]).astype(pos.dtype)
            pot.append(pp)

        forces.zero_()
        for i in range(nsystems):
            spos = pos[i]
            sbox = box[i][paddle.eye(3).astype(paddle.bool)]  # Use only the diagonal

#             print(f"sbos, box shape {sbox.shape, box.shape}")
            # Bonded terms
            # TODO: We are for sure doing duplicate distance calculations here!
            if "bonds" in self.energies and self.par.bonds is not None:
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

                pot[i]["bonds"] += E.sum()
                if explicit_forces:
                    forcevec = bond_unitvec * force_coeff[:, None]
                    forces[i] = paddleindex_add(forces[i], 0, pairs[:, 0], -forcevec)
                    forces[i] = paddleindex_add(forces[i], 0, pairs[:, 1], forcevec)

            if "angles" in self.energies and self.par.angles is not None:
                _, _, r21 = calculate_distances(spos, self.par.angles[:, 0:2], sbox)
                _, _, r23 = calculate_distances(spos, self.par.angles[:, 2:0:-1], sbox)
                E, angle_forces = evaluate_angles(
                    r21, r23, self.par.angle_params, explicit_forces
                )

                pot[i]["angles"] += E.sum()
                if explicit_forces:
                    forces[i] = paddleindex_add(forces[i], 0, self.par.angles[:, 0], angle_forces[0])
                    forces[i] = paddleindex_add(forces[i], 0, self.par.angles[:, 1], angle_forces[1])
                    forces[i] = paddleindex_add(forces[i], 0, self.par.angles[:, 2], angle_forces[2])

            if "dihedrals" in self.energies and self.par.dihedrals is not None:
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

                pot[i]["dihedrals"] += E.sum()
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

                if "lj" in self.energies:
                    E, force_coeff = evaluate_LJ_internal(
                        nb_dist, aa, bb, scnb, None, None, explicit_forces
                    )
                    pot[i]["lj"] += E.sum()
                    if explicit_forces:
                        forcevec = nb_unitvec * force_coeff[:, None]
                        forces[i] = paddleindex_add(forces[i], 0, idx14[:, 0], -forcevec)
                        forces[i] = paddleindex_add(forces[i], 0, idx14[:, 1], forcevec)
                if "electrostatics" in self.energies:
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
                    pot[i]["electrostatics"] += E.sum()
                    if explicit_forces:
#                         print(f"==force line 276 explicit_forces:{explicit_forces} electrostatics len of idx14[:, 0]:{len(idx14[:, 0])}")
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

                pot[i]["impropers"] += E.sum()
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
                        pot[i][v] += E.sum()
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
                        pot[i][v] += E.sum()
                    elif v == "repulsion":
                        E, force_coeff = evaluate_repulsion(
                            nb_dist,
                            ava_idx,
                            self.par.mapped_atom_types,
                            self.par.A,
                            explicit_forces,
                        )
                        pot[i][v] += E.sum()
                    elif v == "repulsioncg":
                        E, force_coeff = evaluate_repulsion_CG(
                            nb_dist,
                            ava_idx,
                            self.par.mapped_atom_types,
                            self.par.B,
                            explicit_forces,
                        )
                        pot[i][v] += E.sum()
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
#                     if pot[i][ene].requires_grad:
                    if not pot[i][ene].stop_gradient:
                        enesum += pot[i][ene]
            forces[:] = -paddle.autograd.grad(
                enesum, pos, only_inputs=True, retain_graph=True
            )[0]
            if returnDetails:
                return pot
            else:
                return [paddle.sum(paddle.cat(list(pp.values()))) for pp in pot]

        if returnDetails:
            return [{k: v.item() for k, v in pp.items()} for pp in pot]
        else:
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
#     print(f"==calculate_distances atom_pos, atom_idx, box:{atom_pos.shape, atom_idx.shape, box.shape}")

    direction_vec = wrap_dist(atom_pos[atom_idx[:, 0]] - atom_pos[atom_idx[:, 1]], box)
#     print(f"==line 423 of forces direction_vec.shape:{direction_vec.shape}")
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
