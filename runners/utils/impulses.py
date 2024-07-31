import os
import pickle
import time

import numpy as np
import torch
import torch_collisions
import torch_scatter
from torch_geometric.data import Batch

from utils.common import add_field_to_pyg_batch
from utils.io import pickle_dump
from utils.defaults import DEFAULTS
from utils.selfcollisions import CollisionHelper, find_close_faces, get_continuous_collisions

IMPULSES_PARTIAL = True
RIZ_TYPE = 4
RIZ_VELO_REWRITE = False
IMPULSE_MASS = True
RIZ_MASS = False
IMPULSE_RESET_PINNED_END = False
IMPULSE_RESET_PINNED_LOOP = False
DEBUG = False



class CollisionSolver:
    def __init__(self, mcfg):
        self.collision_helper = CollisionHelper(mcfg.device)
        self.mcfg = mcfg

    @staticmethod
    def compute_riz_deltas4(riz_ids, pinned_mask, mass, curr_pos, dx):
        V = riz_ids.shape[0]
        device = riz_ids.device
        riz_pinned = pinned_mask[riz_ids]

        masses = mass[riz_ids]
        velocities = dx[riz_ids]
        positions_t0 = curr_pos[riz_ids]

        # 3. Compute center of mass and average velocity
        mass_weighted_positions = masses * positions_t0
        total_mass = torch.sum(masses)
        center_of_mass = torch.sum(mass_weighted_positions, dim=0) / total_mass

        mass_weighted_velocities = masses * velocities
        average_velocity = torch.sum(mass_weighted_velocities, dim=0) / total_mass

        # Compute the inertia tensor
        positions_relative_to_com = positions_t0 - center_of_mass
        # I = torch.zeros(3, 3).to(device).double()
        mass_weighted_square_distances = positions_relative_to_com.pow(2).sum(1) * masses[:, 0]
        eye = torch.eye(3).to(positions_t0.device, positions_t0.dtype)
        mass_weighted_square_distances = mass_weighted_square_distances[:, None, None] * eye[None]
        posmass = torch.einsum(
            'ij,ik->ijk', positions_relative_to_com, positions_relative_to_com) * masses.view(-1, 1, 1)
        I = (mass_weighted_square_distances - posmass).sum(dim=0)

        # Compute the angular momentum
        angular_momentum = torch.cross(mass_weighted_velocities, positions_relative_to_com).sum(dim=0)

        new_angular_velocities = torch.linalg.solve(I.unsqueeze(0), angular_momentum.unsqueeze(0))
        new_angular_velocities = new_angular_velocities.squeeze()  # omega

        angular_velocity_magnitude = torch.norm(new_angular_velocities)
        rotation_axis = new_angular_velocities / angular_velocity_magnitude
        angle = angular_velocity_magnitude

        cos_angle = torch.cos(angle)
        sin_angle = torch.sin(angle)

        dot_product = rotation_axis[None,] * positions_relative_to_com

        term1 = positions_relative_to_com * cos_angle
        term2 = torch.cross(rotation_axis[None,], positions_relative_to_com) * sin_angle
        term3 = rotation_axis[None,] * dot_product * (1 - cos_angle)

        positions_relative_to_com_rotated = term1 + term2 + term3
        new_positions_t1 = center_of_mass + positions_relative_to_com_rotated + average_velocity

        rigid_dx = new_positions_t1 - positions_t0
        return rigid_dx

    @staticmethod
    def compute_riz_deltas3(riz_ids, pinned_mask, mass, curr_pos, dx):
        V = riz_ids.shape[0]
        device = riz_ids.device
        riz_pinned = pinned_mask[riz_ids]

        masses = mass[riz_ids]
        velocities = dx[riz_ids]
        positions_t0 = curr_pos[riz_ids]

        # 3. Compute center of mass and average velocity
        mass_weighted_positions = masses * positions_t0
        total_mass = torch.sum(masses)
        center_of_mass = torch.sum(mass_weighted_positions, dim=0) / total_mass

        mass_weighted_velocities = masses * velocities
        average_velocity = torch.sum(mass_weighted_velocities, dim=0) / total_mass

        # Compute the inertia tensor
        positions_relative_to_com = positions_t0 - center_of_mass
        # I = torch.zeros(3, 3).to(device).double()
        mass_weighted_square_distances = positions_relative_to_com.pow(2).sum(1) * masses[:, 0]
        eye = torch.eye(3).to(positions_t0.device, positions_t0.dtype)
        mass_weighted_square_distances = mass_weighted_square_distances[:, None, None] * eye[None]
        posmass = torch.einsum(
            'ij,ik->ijk', positions_relative_to_com, positions_relative_to_com) * masses.view(-1, 1, 1)
        I = (mass_weighted_square_distances - posmass).sum(dim=0)

        # Compute the angular momentum
        angular_momentum = torch.cross(mass_weighted_velocities, positions_relative_to_com).sum(dim=0)
        new_angular_velocities = torch.linalg.solve(I.unsqueeze(0), angular_momentum.unsqueeze(0))
        new_angular_velocities = new_angular_velocities.squeeze()  # omega

        # Compute the new positions for t=1
        positions_t1_rigid = positions_t0 + average_velocity

        # new_positions_t1 = positions_t1_rigid + pos_fixed + torch.cos(omega_mag) * pos_rot + rot_cross
        positions_relative_to_com_t1 = torch.cross(new_angular_velocities.repeat(V, 1), positions_relative_to_com)
        new_positions_t1 = positions_t1_rigid + positions_relative_to_com_t1

        rigid_dx = new_positions_t1 - positions_t0
        return rigid_dx

    @staticmethod
    def compute_riz_deltas2(riz_ids, pinned_mask, mass, curr_pos, dx):
        V = riz_ids.shape[0]
        device = riz_ids.device
        riz_pinned = pinned_mask[riz_ids]

        masses = mass[riz_ids]
        velocities = dx[riz_ids]
        positions_t0 = curr_pos[riz_ids]

        # 3. Compute center of mass and average velocity
        mass_weighted_positions = masses * positions_t0
        total_mass = torch.sum(masses)
        center_of_mass = torch.sum(mass_weighted_positions, dim=0) / total_mass

        mass_weighted_velocities = masses * velocities
        average_velocity = torch.sum(mass_weighted_velocities, dim=0) / total_mass

        # Compute the inertia tensor
        positions_relative_to_com = positions_t0 - center_of_mass
        I = torch.zeros(3, 3).to(device).double()
        for i in range(V):
            I += masses[i] * (torch.sum(positions_relative_to_com[i] ** 2) * torch.eye(3).to(device) - torch.outer(
                positions_relative_to_com[i], positions_relative_to_com[i]))

        # Compute the angular momentum
        angular_momentum = torch.cross(mass_weighted_velocities, positions_relative_to_com).sum(dim=0)
        new_angular_velocities = torch.linalg.solve(I.unsqueeze(0), angular_momentum.unsqueeze(0))
        new_angular_velocities = new_angular_velocities.squeeze()  # omega

        # Compute the new positions for t=1
        positions_t1_rigid = positions_t0 + average_velocity
        positions_relative_to_com_t1 = torch.cross(new_angular_velocities.repeat(V, 1), positions_relative_to_com)
        new_positions_t1 = positions_t1_rigid + positions_relative_to_com_t1

        rigid_dx = new_positions_t1 - positions_t0
        return rigid_dx

    @staticmethod
    def compute_riz_deltas(riz_ids, pinned_mask, mass, curr_pos, dx):
        N = riz_ids.shape[0]
        device = riz_ids.device
        riz_pinned = pinned_mask[riz_ids]
        riz_mass = mass[riz_ids]
        riz_dx = dx[riz_ids]
        riz_pos = curr_pos[riz_ids]

        center_of_mass = (riz_pos * riz_mass).sum(dim=0) / riz_mass.sum()
        gm = riz_pos - center_of_mass[None]

        if riz_pinned.sum() > 0:
            rigid_dx = (riz_dx[riz_pinned] * riz_mass[riz_pinned]).sum(dim=0) / riz_mass[riz_pinned].sum()
        else:
            rigid_dx = (riz_dx * riz_mass).sum(dim=0) / riz_mass.sum()
        velocity_gm = riz_dx - rigid_dx[None]

        rigid_dx = rigid_dx[None].repeat(N, 1)

        L = torch.cross(gm, velocity_gm, dim=-1) * riz_mass
        L = L.sum(dim=0)
        mag_gm_sq = gm.pow(2).sum(dim=-1, keepdims=True).repeat(1, 3)
        mag_gm_sq_diag = torch.zeros(N, 3, 3).to(device)
        mag_gm_sq_diag = torch.diagonal_scatter(mag_gm_sq_diag, mag_gm_sq, 0, dim1=1, dim2=2)
        gm_outer = torch.bmm(gm[:, :, None], gm[:, None, :])
        Is = (mag_gm_sq_diag - gm_outer) * riz_mass[..., None]  # .sum(dim=0)
        I = Is.sum(dim=0)
        I_det = torch.det(I.cpu()).abs()
        if I_det < 1e-10:
            omega = torch.zeros_like(L)
            pos_fixed = gm
            pos_rot = gm - pos_fixed
            omega_sin = torch.zeros_like(L)
            omega_mag = torch.linalg.norm(omega)
        else:
            omega = torch.linalg.solve(I.cpu(), L[:, None].cpu())[:, 0].to(device)

            omega_mag_old = torch.linalg.norm(omega)
            omega_mag = torch.asin(omega_mag_old)
            omega = omega / omega_mag_old * omega_mag

            pos_fixed = omega[None] * (gm * omega[None]).sum(dim=-1, keepdims=True) / omega.pow(2).sum()
            pos_rot = gm - pos_fixed
            omega_sin = torch.sin(omega_mag) * omega / omega_mag
        omega_mag = torch.linalg.norm(omega)
        omega_sin = omega_sin[None].repeat(N, 1)
        rot_cross = torch.cross(omega_sin, pos_rot)
        total_pos = center_of_mass[None] + rigid_dx + pos_fixed + torch.cos(omega_mag) * pos_rot + rot_cross
        rigid_dx = total_pos - riz_pos
        return rigid_dx

    def check_nan(self, tensor, a=False):
        is_nan = (tensor != tensor).any()
        if a:
            assert not is_nan
        else:
            if is_nan:
                print('NAN')

    def get_faces(self, state, use_cutout=True):
        faces = state['cloth'].faces_batch.T
        if use_cutout and 'faces_cutout_mask_batch' in state['cloth']:
            faces_cutout_mask = state['cloth'].faces_cutout_mask_batch[0]
            faces = faces[faces_cutout_mask]
        return faces

    def safecheck_RIZ(self, state, metrics_dict=None, label=''):
        faces = self.get_faces(state)

        verts0 = state['cloth'].pos.clone()
        verts1 = state['cloth'].pred_pos.clone()
        velocity = state['cloth'].pred_velocity.clone()
        timestep = state['cloth'].timestep


        if self.mcfg.ts_agnostic:
            velocity_dx = velocity * timestep[0]
        else:
            velocity_dx = velocity

        mass = state['cloth'].v_mass.clone()


        # if self.mcfg.double_precision_riz:
        #     verts0 = verts0.double()
        #     verts1 = verts1.double()
        #     velocity = velocity.double()
        #     mass = mass.double()

        dx = verts1 - verts0
        edges = None
        if 'penetrating_mask' in state['cloth']:
            penetrating_mask = state['cloth'].penetrating_mask
            triangles_penetrating = penetrating_mask[faces].unsqueeze(dim=0).contiguous()[0].any(dim=1)
        else:
            triangles_penetrating = None

        vertex_type = state['cloth'].vertex_type.squeeze()
        pinned_mask = vertex_type == 3

        # print('RIZ_MASS', RIZ_MASS)
        if RIZ_MASS:
            if self.mcfg.pinned_mass > 0:
                mass[pinned_mask] = self.mcfg.pinned_mass
        riz_list = []

        iter = 0

        max_riz_size = 0
        while True:
            # print(f'riz iter {iter}')
            collisions_tri = find_close_faces(verts1, faces, threshold=self.mcfg.riz_epsilon)

            if triangles_penetrating is not None:
                collision_penetrating_mask = triangles_penetrating[collisions_tri[:, :2]].any(dim=1)[..., 0]
                collision_nonpenetrating_mask = torch.logical_not(collision_penetrating_mask)
                collisions_tri = collisions_tri[collision_nonpenetrating_mask]

            # print('RIZ', collisions_tri.shape[0])

            if collisions_tri.shape[0] == 0:
                break

            n_edges = 0 if edges is None else edges.shape[0]
            riz_list, edges = self.collision_helper.make_rigid_impact_zones_tritri(faces, collisions_tri,
                                                                                   edges=edges)

            riz_sizes = [x.shape[0] for x in riz_list]
            if len(riz_sizes) > 0:
                max_size = np.max(riz_sizes)
                max_riz_size = max(max_riz_size, max_size)
                if self.mcfg.max_riz_size > 0:
                    if max_size > self.mcfg.max_riz_size:
                        # print('MAX RIZ SIZE EXCEEDED', max_size)
                        break

            if edges.shape[0] == n_edges:
                break

            iter += 1

            for riz in riz_list:

                # print('RIZ_TYPE', RIZ_TYPE)
                if RIZ_TYPE == 1:
                    rigid_dx = self.compute_riz_deltas(riz, pinned_mask, mass, verts0, dx)
                elif RIZ_TYPE == 2:
                    rigid_dx = self.compute_riz_deltas2(riz, pinned_mask, mass, verts0, dx)
                elif RIZ_TYPE == 3:
                    rigid_dx = self.compute_riz_deltas3(riz, pinned_mask, mass, verts0, dx)
                elif RIZ_TYPE == 4:
                    rigid_dx = self.compute_riz_deltas4(riz, pinned_mask, mass, verts0, dx)
                verts1[riz] = verts0[riz] + rigid_dx
                velocity_dx[riz] = rigid_dx

            if iter > self.mcfg.riz_max_steps_total:
                print(f' RIZ reached {self.mcfg.riz_max_steps_total}')
                print('edges', edges.shape[0])

        # print('max_riz_size', max_riz_size)
        if metrics_dict is not None:
            label = label + 'riz_iters'
            metrics_dict[label].append(iter)
            label = label + 'max_riz_size'
            metrics_dict[label].append(max_riz_size)

        if DEBUG:
            print('max_riz_size:', max_riz_size)

        # if self.mcfg.double_precision_riz:
        #     verts1 = verts1.float()
        #     velocity = velocity.float()
        # print('RIZ steps:\t\t', iter)
        state['cloth'].pred_pos = verts1

        if self.mcfg.ts_agnostic:
            velocity = velocity_dx / timestep[0]

        # print('RIZ_VELO_REWRITE', RIZ_VELO_REWRITE)
        if RIZ_VELO_REWRITE:
            state['cloth'].pred_velocity = velocity

        return state

    def impulses_compute(self, state, metrics_dict=None, label=None):

        faces = self.get_faces(state)
        verts0 = state['cloth'].pos.clone()
        verts1 = state['cloth'].pred_pos.clone()
        mass = state['cloth'].v_mass.clone()

        # if self.mcfg.double_precision_impulse:
        #     verts0 = verts0.double()
        #     verts1 = verts1.double()
        #     mass = mass.double()

        if 'penetrating_mask' in state['cloth']:
            penetrating_mask = state['cloth'].penetrating_mask
            triangles_penetrating = penetrating_mask[faces].unsqueeze(dim=0).contiguous()
        else:
            triangles_penetrating = None

        vertex_type = state['cloth'].vertex_type.squeeze()
        pinned_mask = vertex_type == 3

        if IMPULSE_MASS:
            if self.mcfg.pinned_mass > 0:
                mass[pinned_mask] = self.mcfg.pinned_mass

        unpinned_mask = torch.logical_not(pinned_mask)
        unpinned_mask = unpinned_mask[:, None]

        vertex_dx_sum = torch.zeros_like(verts1)
        vertex_dv_sum = torch.zeros_like(verts1)
        triangles_mass = mass[faces].unsqueeze(dim=0).contiguous()

        ncoll = None

        iter = 0
        # print('\n')

        w = 1
        impulsed_points = []

        for i in range(self.mcfg.n_impulse_iters):
            # print(f"step {i}")
            verts1_curr = verts1 + vertex_dx_sum

            triangles1 = verts0[faces].unsqueeze(dim=0).contiguous()
            triangles2 = verts1_curr[faces].unsqueeze(dim=0).contiguous()

            bboxes, tree = torch_collisions.bvh_motion(triangles1, triangles2)
            imp_dv, imp_dx, imp_counter = torch_collisions.compute_impulses(bboxes, tree, triangles1, triangles2,
                                                                            triangles_mass,
                                                                            32*3, 16)
            imp_counter = imp_counter.long()

            if triangles_penetrating is not None:
                imp_counter = imp_counter * torch.logical_not(triangles_penetrating[..., 0])
                imp_dx = imp_dx * torch.logical_not(triangles_penetrating)
                imp_dv = imp_dv * torch.logical_not(triangles_penetrating)

            if ncoll is None:
                ncoll = imp_counter.sum().item() / 4

            if self.mcfg.max_ncoll > 0 and ncoll > self.mcfg.max_ncoll:

                break

            if imp_dv.sum() != imp_dv.sum():
                out_dict = {}
                out_dict['verts0'] = verts0
                out_dict['verts1_curr'] = verts1_curr
                out_dict['faces'] = faces
                out_dict['mass'] = mass

                out_path = os.path.join(DEFAULTS.run_dir, 'debug', 'impulse_nan.pkl')
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                print('IMPULSE NAN SAVED IN ', out_path)
                pickle_dump(out_dict, out_path)

            if imp_counter.sum() == 0:


                # collisions_tri = find_close_faces(verts1_curr, faces, threshold=self.mcfg.riz_epsilon)
                # print('collisions_tri1', collisions_tri.shape)
                # verts1_curr_f = verts1_curr.float()
                # verts1_curr_f = verts1_curr_f.double()
                #
                # collisions_tri = find_close_faces(verts1_curr_f.float().double(), faces, threshold=self.mcfg.riz_epsilon)
                # print('collisions_tri2', collisions_tri.shape)
                #
                # print('verts0', verts0.sum().item())
                # print('verts1_curr', verts1_curr.sum().item())
                # print('faces', faces.sum().item())
                # print('triangles1', triangles1.sum().item())
                # print('triangles2', triangles2.sum().item())
                # print('triangles_mass', triangles_mass.sum().item())
                # print('bboxes', bboxes.sum().item())
                # print('tree', tree.sum().item())
                # print('imp_dv', imp_dv.sum().item())
                # print('imp_counter', imp_counter.sum().item())

                # print('impulses solved!')
                break

            vertex_counts = torch.zeros_like(verts1[:, 0]).long()
            vertex_dx = torch.zeros_like(verts1)
            vertex_dv = torch.zeros_like(verts1)
            vertex_counts = torch_scatter.scatter(imp_counter.reshape(-1), faces.reshape(-1), dim=0, out=vertex_counts)

            vertex_counts = vertex_counts[:, None]
            vertex_counts[vertex_counts == 0] = 1

            torch_scatter.scatter(imp_dx.reshape(-1, 3), faces.reshape(-1), dim=0, out=vertex_dx)
            torch_scatter.scatter(imp_dv.reshape(-1, 3), faces.reshape(-1), dim=0, out=vertex_dv)

            vertex_dx = vertex_dx / vertex_counts
            vertex_dv = vertex_dv / vertex_counts

            vertex_dx = vertex_dx * w
            vertex_dv = vertex_dv * w

            # print('IMPULSE_RESET_PINNED_LOOP', IMPULSE_RESET_PINNED_LOOP)
            if IMPULSE_RESET_PINNED_LOOP:
                vertex_dx = vertex_dx * unpinned_mask
                vertex_dv = vertex_dv * unpinned_mask

            vertex_dx_sum = vertex_dx_sum + vertex_dx
            vertex_dv_sum = vertex_dv_sum + vertex_dv

            iter += 1

        if ncoll is None:
            ncoll = 0

        # print('IMPULSE_RESET_PINNED_END', IMPULSE_RESET_PINNED_END)
        if IMPULSE_RESET_PINNED_END:
            vertex_dx_sum = vertex_dx_sum * unpinned_mask
            vertex_dv_sum = vertex_dv_sum * unpinned_mask

        # print('IMPULSE ITER', iter)
        # if iter == self.mcfg.n_impulse_iters:
        #     print('MAX IMPULSE ITERATIONS REACHED')

        if metrics_dict is not None:
            label_iter = label + 'impulse_iters'
            metrics_dict[label_iter].append(iter)
            label_ncoll = label + 'impulse_stencil_ncoll'
            metrics_dict[label_ncoll].append(ncoll)

        # if self.mcfg.double_precision_impulse:
        #     vertex_dx_sum = vertex_dx_sum.float()
        #     vertex_dv_sum = vertex_dv_sum.float()

        return vertex_dx_sum, vertex_dv_sum


    def impulses_compute_partial(self, state, metrics_dict=None, label=None):

        faces = self.get_faces(state)
        verts0 = state['cloth'].pos.clone()
        verts1 = state['cloth'].pred_pos.clone()
        mass = state['cloth'].v_mass.clone()

        # if self.mcfg.double_precision_impulse:
        #     verts0 = verts0.double()
        #     verts1 = verts1.double()
        #     mass = mass.double()

        if 'penetrating_mask' in state['cloth']:
            penetrating_mask = state['cloth'].penetrating_mask
            triangles_penetrating = penetrating_mask[faces].unsqueeze(dim=0).contiguous()
        else:
            triangles_penetrating = None

        vertex_type = state['cloth'].vertex_type.squeeze()
        pinned_mask = vertex_type == 3

        if IMPULSE_MASS:
            if self.mcfg.pinned_mass > 0:
                mass[pinned_mask] = self.mcfg.pinned_mass

        unpinned_mask = torch.logical_not(pinned_mask)
        unpinned_mask = unpinned_mask[:, None]

        vertex_dx_sum = torch.zeros_like(verts1)
        vertex_dv_sum = torch.zeros_like(verts1)
        triangles_mass = mass[faces].unsqueeze(dim=0).contiguous()

        ncoll = None

        iter = 0
        # print('\n')

        w = 1
        impulsed_points = []
        faces_to_check = None
        for i in range(self.mcfg.n_impulse_iters):
            # print(f"step {i}")
            verts1_curr = verts1 + vertex_dx_sum

            triangles1 = verts0[faces].unsqueeze(dim=0).contiguous()
            triangles2 = verts1_curr[faces].unsqueeze(dim=0).contiguous()

            bboxes, tree = torch_collisions.bvh_motion(triangles1, triangles2)

            if faces_to_check is None:
                imp_dv, imp_dx, imp_counter = torch_collisions.compute_impulses(bboxes, tree, triangles1, triangles2,
                                                                                triangles_mass,
                                                                                32 * 3, 16)
            else:
                imp_dv, imp_dx, imp_counter = torch_collisions.compute_impulses_partial(bboxes, tree, triangles1,
                                                                                        triangles2,
                                                                                        triangles_mass, faces_to_check,
                                                                                        32 * 3, 16)

            imp_counter = imp_counter.long()


            if triangles_penetrating is not None:
                imp_counter = imp_counter * torch.logical_not(triangles_penetrating[..., 0])
                imp_dx = imp_dx * torch.logical_not(triangles_penetrating)
                imp_dv = imp_dv * torch.logical_not(triangles_penetrating)


            if ncoll is None:
                ncoll = imp_counter.sum().item() / 4

            if self.mcfg.max_ncoll > 0 and ncoll > self.mcfg.max_ncoll:

                break

            if imp_dv.sum() != imp_dv.sum():
                out_dict = {}
                out_dict['verts0'] = verts0
                out_dict['verts1_curr'] = verts1_curr
                out_dict['faces'] = faces
                out_dict['mass'] = mass

                out_path = os.path.join(DEFAULTS.run_dir, 'debug', 'impulse_nan.pkl')
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                print('IMPULSE NAN SAVED IN ', out_path)
                pickle_dump(out_dict, out_path)

            if imp_counter.sum() == 0:


                # collisions_tri = find_close_faces(verts1_curr, faces, threshold=self.mcfg.riz_epsilon)
                # print('collisions_tri1', collisions_tri.shape)
                # verts1_curr_f = verts1_curr.float()
                # verts1_curr_f = verts1_curr_f.double()
                #
                # collisions_tri = find_close_faces(verts1_curr_f.float().double(), faces, threshold=self.mcfg.riz_epsilon)
                # print('collisions_tri2', collisions_tri.shape)
                #
                # print('verts0', verts0.sum().item())
                # print('verts1_curr', verts1_curr.sum().item())
                # print('faces', faces.sum().item())
                # print('triangles1', triangles1.sum().item())
                # print('triangles2', triangles2.sum().item())
                # print('triangles_mass', triangles_mass.sum().item())
                # print('bboxes', bboxes.sum().item())
                # print('tree', tree.sum().item())
                # print('imp_dv', imp_dv.sum().item())
                # print('imp_counter', imp_counter.sum().item())

                # print('impulses solved!')
                break

            vertex_dx_sum, vertex_dv_sum, faces_to_check = update_verts(vertex_dx_sum, vertex_dv_sum, verts1, faces,
                                                                        imp_counter, imp_dx, imp_dv, unpinned_mask, w=w)


            iter += 1

        if ncoll is None:
            ncoll = 0

        # print('IMPULSE_RESET_PINNED_END', IMPULSE_RESET_PINNED_END)
        if IMPULSE_RESET_PINNED_END:
            vertex_dx_sum = vertex_dx_sum * unpinned_mask
            vertex_dv_sum = vertex_dv_sum * unpinned_mask

        # print('IMPULSE ITER', iter)
        # if iter == self.mcfg.n_impulse_iters:
        #     print('MAX IMPULSE ITERATIONS REACHED')

        if metrics_dict is not None:
            label_iter = label + 'impulse_iters'
            metrics_dict[label_iter].append(iter)
            label_ncoll = label + 'impulse_stencil_ncoll'
            metrics_dict[label_ncoll].append(ncoll)

        # if self.mcfg.double_precision_impulse:
        #     vertex_dx_sum = vertex_dx_sum.float()
        #     vertex_dv_sum = vertex_dv_sum.float()

        return vertex_dx_sum, vertex_dv_sum



    def safecheck_impulses(self, state, metrics_dict=None, label='', update=True):
        if IMPULSES_PARTIAL:
            vertex_dx_sum, vertex_dv_sum = self.impulses_compute_partial(state, metrics_dict, label)
        else:
            vertex_dx_sum, vertex_dv_sum = self.impulses_compute(state, metrics_dict, label)

        pred_pos = state['cloth'].pred_pos + vertex_dx_sum
        timestep = state['cloth'].timestep

        if self.mcfg.ts_agnostic:
            vertex_dv_sum = vertex_dv_sum / timestep[0]


        pred_velocity = state['cloth'].pred_velocity + vertex_dv_sum

        add_field_to_pyg_batch(state, 'hc_impulse_dx', vertex_dx_sum, 'cloth', 'pos')
        add_field_to_pyg_batch(state, 'hc_impulse_dv', vertex_dv_sum, 'cloth', 'pos')

        if update:
            state['cloth'].pred_pos = pred_pos
            state['cloth'].pred_velocity = pred_velocity
        else:
            add_field_to_pyg_batch(state, 'hc_impulse_pos', pred_pos, 'cloth', 'pos')
            add_field_to_pyg_batch(state, 'hc_impulse_velocity', pred_velocity, 'cloth', 'pos')

        return state

    def compute_impulses_with_interpolation(self, triangles1, triangles2, triangles_mass):

        if self.mcfg.n_ccoll_interpolation == 1:
            bboxes, tree = torch_collisions.bvh_motion(triangles1, triangles2)
            imp_dv, imp_dx, imp_counter = torch_collisions.compute_impulses(bboxes, tree, triangles1, triangles2,
                                                                            triangles_mass,
                                                                            64 * 4, 16)
            return imp_dv, imp_dx, imp_counter

        deltas = triangles2 - triangles1
        ts = torch.linspace(0, 1, self.mcfg.n_ccoll_interpolation + 1).numpy().tolist()

        collisions_list = []
        roots_list = []

        for i in range(self.mcfg.n_ccoll_interpolation):
            tri_from = triangles1 + deltas * ts[i]
            tri_to = triangles1 + deltas * ts[i + 1]
            dts = ts[i + 1] - ts[i]

            bboxes, tree = torch_collisions.bvh_motion(tri_from, tri_to)
            cont_collisions, roots = torch_collisions.find_collisions_continuous(bboxes, tree, tri_from, tri_to, 64,
                                                                                 16)

            mask = cont_collisions[0, :, 0] >= 0
            cont_collisions = cont_collisions[:, mask]
            roots = roots[:, mask]

            roots = ts[i] + roots * dts

            collisions_list.append(cont_collisions)
            roots_list.append(roots)

        collisions_list = torch.cat(collisions_list, dim=1)
        roots_list = torch.cat(roots_list, dim=1)

        if roots_list.shape[1] == 0:
            imp_dv = torch.zeros_like(triangles1)
            imp_dx = torch.zeros_like(triangles1)
            imp_counter = torch.zeros_like(triangles1[..., 0]).long()
            return imp_dv, imp_dx, imp_counter

        imp_dv, imp_dx, imp_counter = torch_collisions.compute_impulses_from_collisions(collisions_list, roots_list,
                                                                                        triangles1,
                                                                                        triangles2, triangles_mass)

        return imp_dv, imp_dx, imp_counter

    def collect_colliding_vertices(self, sequence):
        pos = sequence['cloth'].pos_noglort
        if len(pos.shape) == 3:
            pos = pos[:, 0]
        rest_pos = sequence['cloth'].rest_pos
        faces = sequence['cloth'].faces_batch.T
        device = faces.device

        faces_cont = set()
        N = self.mcfg.n_rest2pos_steps
        for i in range(N + 1):
            vs = rest_pos + (pos - rest_pos) * (i / N)
            ve = rest_pos + (pos - rest_pos) * ((i + 1) / N)
            collision_tensor2, roots = get_continuous_collisions(vs, ve, faces, n_candidates_per_triangle=320)
            faces_cont.update(set(collision_tensor2[:, :2].reshape(-1).cpu().numpy().tolist()))

        collisions_tri = find_close_faces(pos, faces, threshold=0)
        faces_cont.update(set(collisions_tri[:, :2].reshape(-1).cpu().numpy().tolist()))

        faces_penetrating = torch.LongTensor(list(list(faces_cont))).to(device)
        verts_penetrating = faces[faces_penetrating].view(-1)

        penetrating_mask = torch.zeros_like(pos[:, :1]).long()
        penetrating_mask[verts_penetrating] = 1

        sequence = add_field_to_pyg_batch(sequence, 'penetrating_mask', penetrating_mask, 'cloth', reference_key='pos')
        return sequence

    def update_penetrating_mask_cont(self, sequence):
        pred_pos = sequence['cloth'].pred_pos.double()
        pos = sequence['cloth'].pos.double()
        if len(pos.shape) == 3:
            pos = pos[:, 0]
            pred_pos = pred_pos[:, 0]
        faces = sequence['cloth'].faces_batch.T
        device = faces.device
        penetrating_mask_old = sequence['cloth'].penetrating_mask
        triangles_penetrating = penetrating_mask_old[faces].unsqueeze(dim=0).contiguous()[0].any(dim=1)

        faces_cont = set()

        collision_tensor2, roots = get_continuous_collisions(pos, pred_pos, faces, n_candidates_per_triangle=64)
        collision_tensor2 = collision_tensor2[:, :2]
        collision_penetrating_mask = triangles_penetrating[collision_tensor2[:, :2]].any(dim=1)[..., 0]
        collision_nonpenetrating_mask = torch.logical_not(collision_penetrating_mask)
        collision_tensor2 = collision_tensor2[collision_nonpenetrating_mask]
        faces_cont.update(set(collision_tensor2[:, :2].reshape(-1).cpu().numpy().tolist()))

        faces_penetrating = torch.LongTensor(list(list(faces_cont))).to(device)
        verts_penetrating = faces[faces_penetrating].view(-1)
        sequence['cloth'].penetrating_mask[verts_penetrating] = 1
        return sequence

    def update_penetrating_mask_tri(self, sequence):
        pred_pos = sequence['cloth'].pred_pos.double()
        if len(pred_pos.shape) == 3:
            pred_pos = pred_pos[:, 0]
        faces = sequence['cloth'].faces_batch.T
        device = faces.device
        penetrating_mask_old = sequence['cloth'].penetrating_mask
        collisions_tri = find_close_faces(pred_pos, faces, threshold=0)
        faces_cont = set()
        faces_cont.update(set(collisions_tri[:, :2].reshape(-1).cpu().numpy().tolist()))
        faces_penetrating = torch.LongTensor(list(list(faces_cont))).to(device)
        verts_penetrating = faces[faces_penetrating].view(-1)
        sequence['cloth'].penetrating_mask[verts_penetrating] = 1
        return sequence

    def collect_colliding_vertices_from_pos(self, sequence, pred=False, include_pinned=False):
        pos = sequence['cloth'].pred_pos if pred else sequence['cloth'].pos

        if len(pos.shape) == 3:
            pos = pos[:, 0]
        rest_pos = sequence['cloth'].rest_pos
        faces = sequence['cloth'].faces_batch.T
        device = faces.device

        transl = sequence['cloth'].transl
        rotmat = sequence['cloth'].rotmat
        rotmat = torch.inverse(rotmat)

        pos_unposed = pos.clone()
        pos_unposed = pos_unposed - transl
        pos_unposed = torch.mm(pos_unposed, rotmat.permute(1, 0))
        deltas = pos_unposed - rest_pos

        faces_cont = set()
        collisions_tri = find_close_faces(pos_unposed, faces, threshold=0)
        if collisions_tri.shape[0] == 0:
            penetrating_mask = torch.zeros_like(pos[:, :1]).bool()

            if include_pinned:
                vertex_type = sequence['cloth'].vertex_type
                pinned_mask = vertex_type == 3
                penetrating_mask = penetrating_mask + pinned_mask

            sequence = add_field_to_pyg_batch(sequence, 'penetrating_mask', penetrating_mask, 'cloth',
                                              reference_key='pos')
            return sequence
        else:
            faces_cont.update(set(collisions_tri[:, :2].reshape(-1).cpu().numpy().tolist()))

        N = self.mcfg.n_rest2pos_steps
        for i in range(N + 1):
            vs = rest_pos + deltas * (i / N)
            ve = rest_pos + deltas * ((i + 1) / N)
            collision_tensor2, roots = get_continuous_collisions(vs, ve, faces, n_candidates_per_triangle=64)
            faces_cont.update(set(collision_tensor2[:, :2].reshape(-1).cpu().numpy().tolist()))

        faces_penetrating = torch.LongTensor(list(list(faces_cont))).to(device)
        verts_penetrating = faces[faces_penetrating].view(-1)

        penetrating_mask = torch.zeros_like(pos[:, :1]).long()
        penetrating_mask[verts_penetrating] = 1

        if include_pinned:
            vertex_type = sequence['cloth'].vertex_type
            pinned_mask = vertex_type == 3
            penetrating_mask = penetrating_mask + pinned_mask

        sequence = add_field_to_pyg_batch(sequence, 'penetrating_mask', penetrating_mask, 'cloth', reference_key='pos')
        return sequence

    @staticmethod
    def calc_tritri_collisions(sample, prev=False, threshold=0.):
        pos = sample['cloth'].pos if prev else sample['cloth'].pred_pos
        pos = pos.double()

        collisions_tri = find_close_faces(pos, sample['cloth'].faces_batch.T, threshold=threshold)

        if 'penetrating_mask' in sample['cloth']:
            faces = sample['cloth'].faces_batch.T

            penetrating_mask = sample['cloth'].penetrating_mask
            triangles_penetrating = penetrating_mask[faces].unsqueeze(dim=0).contiguous()[0].any(dim=1)

            collision_penetrating_mask = triangles_penetrating[collisions_tri[:, :2]].any(dim=1)[..., 0]
            collision_nonpenetrating_mask = torch.logical_not(collision_penetrating_mask)
            collisions_tri = collisions_tri[collision_nonpenetrating_mask]

        if 'faces_cutout_mask_batch' in sample['cloth']:
            faces_mask = sample['cloth'].faces_cutout_mask_batch[0]
            collision_mask = faces_mask[collisions_tri].all(dim=-1)
            collisions_tri = collisions_tri[collision_mask]

        return collisions_tri.shape[0]


    @staticmethod
    def calc_tritri_collisions2(sample, obj_key='cloth', verts_key='pred_pos', threshold=0.):
        pos = sample[obj_key][verts_key]
        faces = sample[obj_key].faces_batch.T
        pos = pos.double()

        collisions_tri = find_close_faces(pos, faces, threshold=threshold)

        if 'penetrating_mask' in sample[obj_key]:
            penetrating_mask = sample[obj_key].penetrating_mask
            triangles_penetrating = penetrating_mask[faces].unsqueeze(dim=0).contiguous()[0].any(dim=1)

            collision_penetrating_mask = triangles_penetrating[collisions_tri[:, :2]].any(dim=1)[..., 0]
            collision_nonpenetrating_mask = torch.logical_not(collision_penetrating_mask)
            collisions_tri = collisions_tri[collision_nonpenetrating_mask]

        if 'faces_cutout_mask_batch' in sample[obj_key]:
            faces_mask = sample[obj_key].faces_cutout_mask_batch[0]
            collision_mask = faces_mask[collisions_tri].all(dim=-1)
            collisions_tri = collisions_tri[collision_mask]

        return collisions_tri.shape[0]


    @staticmethod
    def mark_penetrating_faces(sample, threshold=0., dummy=False, object='cloth', update=False):

        B = sample.num_graphs
        new_examples = []
        for i in range(B):
            example = sample.get_example(i)

            faces = example[object].faces_batch.T
            pos = example[object].pos
            # print('dummy', dummy)
            # print('pos', pos.shape)

            if len(pos.shape) == 3:
                pos = pos[:, 0]

            if dummy:
                node_mask = torch.ones_like(pos[:, 0]).bool()
                faces_mask = torch.ones_like(faces[:, :1]).bool()
                example[object].cutout_mask = node_mask
                example[object].faces_cutout_mask_batch = faces_mask.T
                new_examples.append(example)

                continue

            collisions_tri = find_close_faces(pos, faces, threshold=threshold)
            unique_faces = torch.unique(collisions_tri[:, :2])
            faces_mask = torch.ones_like(faces[:, 0]).bool()
            faces_mask[unique_faces] = 0

            if update:
                faces_mask_prev = example[object].faces_cutout_mask_batch[0]
                faces_mask = torch.logical_and(faces_mask, faces_mask_prev)

            faces_enabled = faces[faces_mask]
            node_ids = torch.unique(faces_enabled)
            node_mask = torch.zeros_like(pos[:, 0]).bool()
            node_mask[node_ids] = 1
            faces_mask = faces_mask[None]

            example[object].cutout_mask = node_mask
            example[object].faces_cutout_mask_batch = faces_mask

            new_examples.append(example)
        sample_new = Batch.from_data_list(new_examples)
        return sample_new


    @staticmethod
    def mark_penetrating_faces_obstacle(sample, threshold=0):
        # return sample

        if 'obstacle' not in sample.node_types:
            return sample

        B = sample.num_graphs
        new_examples = []
        for i in range(B):
            example = sample.get_example(i)

            faces = example['obstacle'].faces_batch.T
            pos = example['obstacle'].pos
            target_pos = example['obstacle'].target_pos
            # print('dummy', dummy)
            # print('pos', pos.shape)

            if len(pos.shape) == 3:
                pos = pos[:, 0]

            faces_mask = torch.ones_like(faces[:, 0]).bool()

            collisions_tri_curr = find_close_faces(pos, faces, threshold=threshold)
            unique_faces_curr = torch.unique(collisions_tri_curr[:, :2])
            faces_mask[unique_faces_curr] = 0

            collisions_tri_next = find_close_faces(target_pos, faces, threshold=threshold)
            unique_faces_next = torch.unique(collisions_tri_next[:, :2])
            faces_mask[unique_faces_next] = 0

            faces_enabled = faces[faces_mask]
            node_ids = torch.unique(faces_enabled)
            node_mask = torch.zeros_like(pos[:, 0]).bool()
            node_mask[node_ids] = 1
            faces_mask = faces_mask[None]

            example['obstacle'].cutout_mask = node_mask
            example['obstacle'].faces_cutout_mask_batch = faces_mask

            new_examples.append(example)
        sample_new = Batch.from_data_list(new_examples)
        return sample_new



def calc_tritri_collisions(faces, pos, penetrating_mask=None, threshold=0., unique=False):
    pos = pos.clone().double()

    collisions_tri = find_close_faces(pos, faces, threshold=threshold)
    if penetrating_mask is None:
        return collisions_tri.shape[0]

    triangles_penetrating = penetrating_mask[faces].unsqueeze(dim=0).contiguous()[0].any(dim=1)

    collision_penetrating_mask = triangles_penetrating[collisions_tri[:, :2]].any(dim=1)[..., 0]
    collision_nonpenetrating_mask = torch.logical_not(collision_penetrating_mask)
    collisions_tri = collisions_tri[collision_nonpenetrating_mask]

    if unique:
        # print(torch.unique(collisions_tri[:, :2]))
        return torch.unique(collisions_tri[:, :2]).shape[0]
    return collisions_tri.shape[0]


def safecheck_impulses_pp(faces, verts0, verts1, velocity, mass, penetrating_mask, vertex_type, pinned_mass=1000,
                          n_impulse_iters=10, w=2, seq=None):
    verts0 = verts0.clone().double()
    verts1 = verts1.clone().double()
    mass = mass.clone().double()

    vertex_type = vertex_type.squeeze()
    pinned_mask = vertex_type == 3

    triangles_penetrating = penetrating_mask[faces].unsqueeze(dim=0).contiguous()[0].any(dim=1)

    if pinned_mass > 0:
        mass[pinned_mask] = pinned_mass

    unpinned_mask = torch.logical_not(pinned_mask)
    unpinned_mask = unpinned_mask[:, None]

    vertex_dx_sum = torch.zeros_like(verts1)
    vertex_dv_sum = torch.zeros_like(verts1)
    triangles_mass = mass[faces].unsqueeze(dim=0).contiguous()

    ncoll = None

    from time import sleep

    iter = 0
    for i in range(n_impulse_iters):
        verts1_curr = verts1 + vertex_dx_sum

        if seq is not None:
            seq.append(verts1_curr.cpu().numpy())

        triangles1 = verts0[faces].unsqueeze(dim=0).contiguous()
        triangles2 = verts1_curr[faces].unsqueeze(dim=0).contiguous()

        bboxes, tree = torch_collisions.bvh_motion(triangles1, triangles2)
        cont_collisions, roots = torch_collisions.find_collisions_continuous(bboxes, tree, triangles1, triangles2,
                                                                             64 * 4,
                                                                             16)
        cont_collisions = cont_collisions[0]
        collision_penetrating_mask = triangles_penetrating[cont_collisions[:, :2]].any(dim=1)[..., 0]
        collision_nonpenetrating_mask = torch.logical_not(collision_penetrating_mask)

        cont_collisions = cont_collisions[collision_nonpenetrating_mask]
        cont_collisions = cont_collisions[None,]

        imp_dv, imp_dx, imp_counter = torch_collisions.compute_impulses_from_collisions(cont_collisions, roots,
                                                                                        triangles1,
                                                                                        triangles2, triangles_mass)

        collision_tensor, roots = get_continuous_collisions(verts0, verts1_curr, faces, n_candidates_per_triangle=160)

        imp_counter = imp_counter.long()

        if ncoll is None:
            ncoll = imp_counter.sum().item()

        if imp_counter.sum() == 0:
            break

        vertex_counts = torch.zeros_like(verts1[:, 0]).long()
        vertex_dx = torch.zeros_like(verts1)
        vertex_dv = torch.zeros_like(verts1)
        vertex_counts = torch_scatter.scatter(imp_counter.reshape(-1), faces.reshape(-1), dim=0, out=vertex_counts)

        vertex_counts = vertex_counts[:, None]
        vertex_counts[vertex_counts == 0] = 1

        torch_scatter.scatter(imp_dx.reshape(-1, 3), faces.reshape(-1), dim=0, out=vertex_dx)
        torch_scatter.scatter(imp_dv.reshape(-1, 3), faces.reshape(-1), dim=0, out=vertex_dv)

        vertex_dx = vertex_dx / vertex_counts
        vertex_dv = vertex_dv / vertex_counts

        vertex_dx = vertex_dx * w
        vertex_dv = vertex_dv * w

        # vertex_dx = vertex_dx * unpinned_mask
        # vertex_dv = vertex_dv * unpinned_mask
        vertex_dx_sum = vertex_dx_sum + vertex_dx
        vertex_dv_sum = vertex_dv_sum + vertex_dv

    vertex_dx_sum = vertex_dx_sum.float()
    vertex_dv_sum = vertex_dv_sum.float()

    pred_pos = verts1 + vertex_dx_sum

    if velocity is None:
        pred_velocity = None
    else:
        pred_velocity = velocity + vertex_dv_sum
    return pred_pos, pred_velocity



def safecheck_impulses(faces, verts0, verts1, velocity, mass,
                       n_impulse_iters=100):
    verts0 = verts0.clone().double()
    verts1 = verts1.clone().double()
    mass = mass.clone().double()


    vertex_dx_sum = torch.zeros_like(verts1)
    vertex_dv_sum = torch.zeros_like(verts1)
    triangles_mass = mass[faces].unsqueeze(dim=0).contiguous()

    ncoll = None

    iter = 0
    # print('\n')

    w = 1
    impulsed_points = []

    for i in range(n_impulse_iters):
        # print(f"step {i}")
        verts1_curr = verts1 + vertex_dx_sum

        triangles1 = verts0[faces].unsqueeze(dim=0).contiguous()
        triangles2 = verts1_curr[faces].unsqueeze(dim=0).contiguous()

        bboxes, tree = torch_collisions.bvh_motion(triangles1, triangles2)
        imp_dv, imp_dx, imp_counter = torch_collisions.compute_impulses(bboxes, tree, triangles1, triangles2,
                                                                        triangles_mass,
                                                                        32 * 3, 16)
        imp_counter = imp_counter.long()

        print('imp_counter', imp_counter.sum().item())

        if ncoll is None:
            ncoll = imp_counter.sum().item() / 4

        if imp_dv.sum() != imp_dv.sum():
            out_dict = {}
            out_dict['verts0'] = verts0
            out_dict['verts1_curr'] = verts1_curr
            out_dict['faces'] = faces
            out_dict['mass'] = mass

            out_path = os.path.join(DEFAULTS.run_dir, 'debug', 'impulse_nan.pkl')
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            print('IMPULSE NAN SAVED IN ', out_path)
            pickle_dump(out_dict, out_path)

        if imp_counter.sum() == 0:
            break

        vertex_counts = torch.zeros_like(verts1[:, 0]).long()
        vertex_dx = torch.zeros_like(verts1)
        vertex_dv = torch.zeros_like(verts1)
        vertex_counts = torch_scatter.scatter(imp_counter.reshape(-1), faces.reshape(-1), dim=0, out=vertex_counts)

        vertex_counts = vertex_counts[:, None]
        vertex_counts[vertex_counts == 0] = 1

        torch_scatter.scatter(imp_dx.reshape(-1, 3), faces.reshape(-1), dim=0, out=vertex_dx)
        torch_scatter.scatter(imp_dv.reshape(-1, 3), faces.reshape(-1), dim=0, out=vertex_dv)

        vertex_dx = vertex_dx / vertex_counts
        vertex_dv = vertex_dv / vertex_counts

        vertex_dx = vertex_dx * w
        vertex_dv = vertex_dv * w

        vertex_dx_sum = vertex_dx_sum + vertex_dx
        vertex_dv_sum = vertex_dv_sum + vertex_dv

        iter += 1

    if ncoll is None:
        ncoll = 0


    vertex_dx_sum = vertex_dx_sum.float()
    vertex_dv_sum = vertex_dv_sum.float()

    pred_pos = verts1 + vertex_dx_sum

    if velocity is None:
        pred_velocity = None
    else:
        pred_velocity = velocity + vertex_dv_sum
    return pred_pos, pred_velocity



def safecheck_impulses_partial(faces, verts0, verts1, velocity, mass,
                       n_impulse_iters=100):
    verts0 = verts0.clone().double()
    verts1 = verts1.clone().double()
    mass = mass.clone().double()


    vertex_dx_sum = torch.zeros_like(verts1)
    vertex_dv_sum = torch.zeros_like(verts1)
    triangles_mass = mass[faces].unsqueeze(dim=0).contiguous()

    ncoll = None

    iter = 0
    # print('\n')

    w = 1
    impulsed_points = []

    faces_to_check = None

    for i in range(n_impulse_iters):
        # print()
        # print(f"step {i}")
        verts1_curr = verts1 + vertex_dx_sum

        triangles1 = verts0[faces].unsqueeze(dim=0).contiguous()
        triangles2 = verts1_curr[faces].unsqueeze(dim=0).contiguous()

        bboxes, tree = torch_collisions.bvh_motion(triangles1, triangles2)

        if faces_to_check is None:
            imp_dv, imp_dx, imp_counter = torch_collisions.compute_impulses(bboxes, tree, triangles1, triangles2,
                                                                            triangles_mass,
                                                                            32 * 3, 16)

        else:
            imp_dv, imp_dx, imp_counter = torch_collisions.compute_impulses_partial(bboxes, tree, triangles1, triangles2,
                                                                            triangles_mass, faces_to_check,
                                                                            32 * 3, 16)

        # out_collisions, roots = torch_collisions.find_collisions_continuous(bboxes, tree, triangles1, triangles2,
        #                                                                     32, 16)
        # out_collisions = out_collisions[0]
        # roots = roots[0]
        # mask = roots[:, 0] >= 0
        # roots = roots[mask]
        # out_collisions = out_collisions[mask]
        # print('out_collisions', out_collisions.shape)
        # colliding_triangles_ids = torch.unique(out_collisions[:, :2])
        # colliding_triangles_mask = torch.zeros_like(faces[:, :1])
        # colliding_triangles_mask[colliding_triangles_ids] = 1
        # colliding_triangles_mask = colliding_triangles_mask.bool()
        # # print('colliding_triangles_mask', colliding_triangles_mask.sum().item())

        # if faces_to_check is not None:
        #     out_collisions, roots = torch_collisions.find_collisions_continuous_partial(bboxes, tree, triangles1, triangles2,
        #                                                                                 faces_to_check, 32, 16)
        #     out_collisions = out_collisions[0]
        #     roots = roots[0]
        #     mask = roots[:, 0] >= 0
        #     roots = roots[mask]
        #     out_collisions = out_collisions[mask]
        #     print('out_collisions_partial', out_collisions.shape)
        #     colliding_triangles_ids = torch.unique(out_collisions[:, :2])
        #     colliding_triangles_mask = torch.zeros_like(faces[:, :1])
        #     colliding_triangles_mask[colliding_triangles_ids] = 1
        #     colliding_triangles_mask = colliding_triangles_mask.bool()
            # print('colliding_triangles_mask partial', colliding_triangles_mask.sum().item())


        imp_counter = imp_counter.long()
        print('imp_counter', imp_counter.sum().item())

        # faces_to_check_ids,  = torch.where(imp_counter[0].sum(dim=-1) > 0)
        # faces_to_check = torch.zeros_like(imp_counter[0, :, :1]).bool()
        # faces_to_check[faces_to_check_ids] = True
        # faces_to_check = faces_to_check[None]
        # faces_to_check = colliding_triangles_mask[None]

        # print('colliding_triangles_mask', colliding_triangles_mask.sum().item())
        # print('colliding_triangles_mask', colliding_triangles_mask.shape)


        if ncoll is None:
            ncoll = imp_counter.sum().item() / 4

        if imp_dv.sum() != imp_dv.sum():
            out_dict = {}
            out_dict['verts0'] = verts0
            out_dict['verts1_curr'] = verts1_curr
            out_dict['faces'] = faces
            out_dict['mass'] = mass

            out_path = os.path.join(DEFAULTS.run_dir, 'debug', 'impulse_nan.pkl')
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            print('IMPULSE NAN SAVED IN ', out_path)
            pickle_dump(out_dict, out_path)

        if imp_counter.sum() == 0:
            break

        vertex_dx_sum, vertex_dv_sum, faces_to_check = update_verts(vertex_dx_sum, vertex_dv_sum, verts1, faces,  imp_counter, imp_dx, imp_dv)

        # print('faces_to_check', faces_to_check.sum().item())
        iter += 1

    if ncoll is None:
        ncoll = 0


    vertex_dx_sum = vertex_dx_sum.float()
    vertex_dv_sum = vertex_dv_sum.float()

    pred_pos = verts1 + vertex_dx_sum

    if velocity is None:
        pred_velocity = None
    else:
        pred_velocity = velocity + vertex_dv_sum
    return pred_pos, pred_velocity



def update_verts(vertex_dx_sum, vertex_dv_sum, verts1, faces, imp_counter, imp_dx, imp_dv,unpinned_mask=None, w=1.):
    vertex_counts = torch.zeros_like(verts1[:, 0]).long()
    vertex_dx = torch.zeros_like(verts1)
    vertex_dv = torch.zeros_like(verts1)

    vertex_counts = torch_scatter.scatter(imp_counter.reshape(-1), faces.reshape(-1), dim=0, out=vertex_counts)
    vertex_changed = vertex_counts > 0
    faces_changed = vertex_changed[faces].any(dim=-1, keepdim=True)

    vertex_counts = vertex_counts[:, None]
    vertex_counts[vertex_counts == 0] = 1

    torch_scatter.scatter(imp_dx.reshape(-1, 3), faces.reshape(-1), dim=0, out=vertex_dx)
    torch_scatter.scatter(imp_dv.reshape(-1, 3), faces.reshape(-1), dim=0, out=vertex_dv)

    vertex_dx = vertex_dx / vertex_counts
    vertex_dv = vertex_dv / vertex_counts

    vertex_dx = vertex_dx * w
    vertex_dv = vertex_dv * w

    if unpinned_mask is not None and IMPULSE_RESET_PINNED_LOOP:
        vertex_dx = vertex_dx * unpinned_mask
        vertex_dv = vertex_dv * unpinned_mask

    vertex_dx_sum = vertex_dx_sum + vertex_dx
    vertex_dv_sum = vertex_dv_sum + vertex_dv

    return vertex_dx_sum, vertex_dv_sum, faces_changed