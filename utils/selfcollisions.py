import pickle

import networkx as nx
import numpy as np
import torch
import torch_collisions as collisions
# from mesh_intersection.bvh_search_tree import BVH
from pytorch3d.ops import knn_points
from torch_scatter import scatter_min
from tqdm import tqdm

from utils.common import gather
from utils.cub_solver import solve_torch


def bdot(v0, v1, dim=-1, keepdim=False):
    d = (v0 * v1).sum(dim=dim, keepdim=keepdim)
    return d


def get_barycoords(points, triangles):
    # print('triangles', triangles.shape)
    # print('points', points.shape)

    v0, v1, v2 = torch.split(triangles, 1, 1)

    e1 = v0 - v1
    e2 = v2 - v1
    e1 = e1[:, 0]
    e2 = e2[:, 0]
    ep = points - v1[:, 0]

    d00 = bdot(e1, e1)
    d01 = bdot(e1, e2)
    d11 = bdot(e2, e2)
    d20 = bdot(ep, e1)  # ?
    d21 = bdot(ep, e2)  # ?
    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1 - v - w

    barycoords = torch.stack([v, u, w], dim=-1)
    return barycoords


def find_close_faces(vertices, faces, threshold=0):
    """
    vertices: Vx3
    faces: Fx3

    return: pairs of (almost) colliding faces Nx2
    """
    # threshold = 0
    triangles = vertices[faces].unsqueeze(dim=0).contiguous()
    bboxes, tree = collisions.bvh(triangles)
    face_pairs = collisions.find_collisions(bboxes, tree, triangles, threshold, 320)[0]
    face_pairs = face_pairs[face_pairs[:, 0] >= 0, :]

    return face_pairs


def find_node_face_pairs(vertices, faces, f_normals, threshold_triangles=3e-2, threshold_normals=3e-2):
    """

    :param vertices: Vx3
    :param faces: Fx3
    :param f_normals:
    :param threshold_triangles:
    :param threshold_normals:
    :return: points_from (N), faces_to (N), barycoords_masked (Nx3), dists_normal_masked (N)
    """
    face_normals = f_normals(vertices.unsqueeze(0), faces.unsqueeze(0))
    triangles = vertices[faces].unsqueeze(dim=0)
    face_pairs = find_close_faces(vertices, faces, threshold=threshold_triangles)

    face_idx_from = face_pairs.reshape(-1)
    face_idx_to = face_pairs[..., [1, 0]].reshape(-1)

    points_from, points_to, faces_to = [], [], []
    for i in range(3):
        points_from.append(faces[face_idx_from, i])
        points_to.append(faces[face_idx_to, 2])
        faces_to.append(face_idx_to)

    points_from = torch.cat(points_from, dim=0)
    points_to = torch.cat(points_to, dim=0)
    faces_to = torch.cat(faces_to, dim=0)

    pos_from = vertices[points_from]
    pos_to = vertices[points_to]
    normals_to = face_normals[0, faces_to]
    triangles_to = triangles[0, faces_to]

    relative_pos = pos_from - pos_to
    relative_pos_norm_proj = relative_pos * normals_to
    points_proj = pos_from - relative_pos_norm_proj
    barycoords = get_barycoords(points_proj, triangles_to)

    barycoords_mask = (barycoords > 1e-7).all(dim=-1)
    dists_normal = relative_pos_norm_proj.sum(-1)
    dists_mask = dists_normal.abs() < threshold_normals
    all_mask = dists_mask * barycoords_mask

    dists_normal_masked = dists_normal[all_mask]
    barycoords_masked = barycoords[all_mask]
    points_from = points_from[all_mask]
    faces_to = faces_to[all_mask]

    return points_from, faces_to, barycoords_masked, dists_normal_masked


def find_node_face_pairs_cuda(vertices, faces, threshold=1e-3):
    """

    :param vertices: Vx3
    :param faces: Fx3
    :param threshold:
    :return: points_from (N), faces_to (N)
    """

    triangles1 = vertices[faces].contiguous()
    bboxes, tree = collisions.bvh(triangles1.unsqueeze(0))
    proximity_tensor = collisions.find_proximity(bboxes, tree, triangles1, threshold, 64)

    proximity_tensor = proximity_tensor[0][0]
    mask = proximity_tensor[:, 0] >= 0
    proximity_tensor = proximity_tensor[mask, :]

    points_from = []
    faces_to = []

    tri_node_mask = torch.logical_and(proximity_tensor[:, 2] >= 9, proximity_tensor[:, 2] < 12)
    faces_to_curr = proximity_tensor[tri_node_mask, 0]
    faces_from_curr = proximity_tensor[tri_node_mask, 1]
    points_from_curr = faces[faces_from_curr]
    v_ids_curr = proximity_tensor[tri_node_mask, 2] - 9
    v_ids_curr = v_ids_curr.unsqueeze(-1)
    points_from_curr = torch.gather(points_from_curr, 1, v_ids_curr)

    points_from.append(points_from_curr)
    faces_to.append(faces_to_curr)

    node_tri_mask = proximity_tensor[:, 2] >= 12
    faces_to_curr = proximity_tensor[node_tri_mask, 1]
    faces_from_curr = proximity_tensor[node_tri_mask, 0]
    points_from_curr = faces[faces_from_curr]
    v_ids_curr = proximity_tensor[node_tri_mask, 2] - 12
    v_ids_curr = v_ids_curr.unsqueeze(-1)
    points_from_curr = torch.gather(points_from_curr, 1, v_ids_curr)

    points_from.append(points_from_curr)
    faces_to.append(faces_to_curr)

    points_from = torch.cat(points_from)[:, 0]
    faces_to = torch.cat(faces_to)

    return points_from, faces_to


def get_node2face_signed_distance(vertices, faces, f_normals, points_from, faces_to, detach_faces=False):
    """

    :param vertices:  Vx3
    :param faces: Fx3
    :param f_normals:
    :param points_from: N
    :param faces_to: N
    :return: dists_normal (N)
    """
    face_normals = f_normals(vertices.unsqueeze(0), faces.unsqueeze(0))

    points_to = faces[faces_to, 2]

    normals_to = face_normals[0, faces_to]
    pos_from = vertices[points_from]
    pos_to = vertices[points_to]

    if detach_faces:
        pos_to = pos_to.detach()

    relative_pos = pos_from - pos_to
    relative_pos_norm_proj = relative_pos * normals_to
    dists_normal = relative_pos_norm_proj.sum(-1)

    return dists_normal


def get_node2node_correspondences(faces, points_from, faces_to, barycoords, all_wedges=False):
    """

    :param faces: Fx3
    :param points_from: N
    :param faces_to:  N
    :param barycoords: Nx3
    :return: Nx2
    """

    points_to = faces[faces_to]
    if all_wedges:
        points_to_list = points_to.split(1, dim=-1)
        points_to = torch.cat(points_to_list, dim=0)[..., 0]
        points_from = points_from.repeat(3)
    else:
        min_ind = torch.argmin(barycoords, dim=-1).unsqueeze(-1)
        points_to = points_to.gather(-1, min_ind)[..., 0]

    edges = torch.stack([points_from, points_to], dim=-1)
    return edges


def make_type2ind():
    type2ind = []
    type2ind.append([0, 1, 3, 4])
    type2ind.append([0, 1, 3, 5])
    type2ind.append([0, 1, 4, 5])
    type2ind.append([0, 2, 3, 4])
    type2ind.append([0, 2, 3, 5])
    type2ind.append([0, 2, 4, 5])
    type2ind.append([1, 2, 3, 4])
    type2ind.append([1, 2, 3, 5])
    type2ind.append([1, 2, 4, 5])

    type2ind.append([0, 1, 2, 3])
    type2ind.append([0, 1, 2, 4])
    type2ind.append([0, 1, 2, 5])

    type2ind.append([0, 3, 4, 5])
    type2ind.append([1, 3, 4, 5])
    type2ind.append([2, 3, 4, 5])

    type2ind = torch.LongTensor(type2ind)
    return type2ind


class ContCollPt:
    def __init__(self):
        self.type2ind = make_type2ind().cuda()

    def coef3(self, mv, md):
        coef = 0
        coef += md[:, 2, 2] * md[:, 1, 1] * md[:, 0, 0]
        coef += - md[:, 2, 1] * md[:, 1, 2] * md[:, 0, 0]
        coef += - md[:, 2, 2] * md[:, 1, 0] * md[:, 0, 1]
        coef += md[:, 2, 0] * md[:, 1, 2] * md[:, 0, 1]
        coef += md[:, 2, 1] * md[:, 1, 0] * md[:, 0, 2]
        coef += - md[:, 2, 0] * md[:, 1, 1] * md[:, 0, 2]
        return coef

    def coef2(self, mv, md):
        coef = 0
        coef += mv[:, 2, 2] * md[:, 1, 1] * md[:, 0, 0]
        coef += md[:, 2, 2] * mv[:, 1, 1] * md[:, 0, 0]
        coef += md[:, 2, 2] * md[:, 1, 1] * mv[:, 0, 0]
        coef += - mv[:, 2, 1] * md[:, 1, 2] * md[:, 0, 0]
        coef += - md[:, 2, 1] * mv[:, 1, 2] * md[:, 0, 0]
        coef += - md[:, 2, 1] * md[:, 1, 2] * mv[:, 0, 0]
        coef += - mv[:, 2, 2] * md[:, 1, 0] * md[:, 0, 1]
        coef += - md[:, 2, 2] * mv[:, 1, 0] * md[:, 0, 1]
        coef += - md[:, 2, 2] * md[:, 1, 0] * mv[:, 0, 1]
        coef += mv[:, 2, 0] * md[:, 1, 2] * md[:, 0, 1]
        coef += md[:, 2, 0] * mv[:, 1, 2] * md[:, 0, 1]
        coef += md[:, 2, 0] * md[:, 1, 2] * mv[:, 0, 1]
        coef += mv[:, 2, 1] * md[:, 1, 0] * md[:, 0, 2]
        coef += md[:, 2, 1] * mv[:, 1, 0] * md[:, 0, 2]
        coef += md[:, 2, 1] * md[:, 1, 0] * mv[:, 0, 2]
        coef += - mv[:, 2, 0] * md[:, 1, 1] * md[:, 0, 2]
        coef += - md[:, 2, 0] * mv[:, 1, 1] * md[:, 0, 2]
        coef += - md[:, 2, 0] * md[:, 1, 1] * mv[:, 0, 2]
        return coef

    def coef1(self, mv, md):
        coef = 0
        coef += mv[:, 2, 2] * mv[:, 1, 1] * md[:, 0, 0]
        coef += mv[:, 2, 2] * md[:, 1, 1] * mv[:, 0, 0]
        coef += md[:, 2, 2] * mv[:, 1, 1] * mv[:, 0, 0]
        coef += - mv[:, 2, 1] * mv[:, 1, 2] * md[:, 0, 0]
        coef += - mv[:, 2, 1] * md[:, 1, 2] * mv[:, 0, 0]
        coef += - md[:, 2, 1] * mv[:, 1, 2] * mv[:, 0, 0]
        coef += - mv[:, 2, 2] * mv[:, 1, 0] * md[:, 0, 1]
        coef += - mv[:, 2, 2] * md[:, 1, 0] * mv[:, 0, 1]
        coef += - md[:, 2, 2] * mv[:, 1, 0] * mv[:, 0, 1]
        coef += mv[:, 2, 0] * mv[:, 1, 2] * md[:, 0, 1]
        coef += mv[:, 2, 0] * md[:, 1, 2] * mv[:, 0, 1]
        coef += md[:, 2, 0] * mv[:, 1, 2] * mv[:, 0, 1]
        coef += mv[:, 2, 1] * mv[:, 1, 0] * md[:, 0, 2]
        coef += mv[:, 2, 1] * md[:, 1, 0] * mv[:, 0, 2]
        coef += md[:, 2, 1] * mv[:, 1, 0] * mv[:, 0, 2]
        coef += - mv[:, 2, 0] * mv[:, 1, 1] * md[:, 0, 2]
        coef += - mv[:, 2, 0] * md[:, 1, 1] * mv[:, 0, 2]
        coef += - md[:, 2, 0] * mv[:, 1, 1] * mv[:, 0, 2]
        return coef

    def coef0(self, mv, md):
        coef = 0
        coef += mv[:, 2, 2] * mv[:, 1, 1] * mv[:, 0, 0]
        coef += - mv[:, 2, 1] * mv[:, 1, 2] * mv[:, 0, 0]
        coef += - mv[:, 2, 2] * mv[:, 1, 0] * mv[:, 0, 1]
        coef += mv[:, 2, 0] * mv[:, 1, 2] * mv[:, 0, 1]
        coef += mv[:, 2, 1] * mv[:, 1, 0] * mv[:, 0, 2]
        coef += - mv[:, 2, 0] * mv[:, 1, 1] * mv[:, 0, 2]
        return coef

    def check_intersects_ee(self, tetrahs):
        da = tetrahs[:, :, 1] - tetrahs[:, :, 0]
        db = tetrahs[:, :, 3] - tetrahs[:, :, 2]
        dc = tetrahs[:, :, 2] - tetrahs[:, :, 0]

        dcb = torch.cross(dc, db)
        dab = torch.cross(da, db)
        s = (dab * dcb).sum(-1) / dab.pow(2).sum(-1)

        dca = torch.cross(dc, da)
        t = (dab * dca).sum(-1) / dab.pow(2).sum(-1)

        s_true = torch.logical_and(s >= 0, s <= 1)
        t_true = torch.logical_and(t >= 0, t <= 1)
        intersects = torch.logical_and(s_true, t_true)
        return intersects

    def check_intersects_tv(self, triangles, points):
        barycoords = get_barycoords(points, triangles)
        mask_u = torch.logical_and(barycoords[:, 0] >= 0, barycoords[:, 0] <= 1)
        mask_v = torch.logical_and(barycoords[:, 1] >= 0, barycoords[:, 1] <= 1)
        mask_w = torch.logical_and(barycoords[:, 2] >= 0, barycoords[:, 2] <= 1)

        inside = torch.logical_and(mask_u, mask_v)
        inside = torch.logical_and(inside, mask_w)

        return inside

    def check_intersects(self, tetrahs, ctype):
        intersects = torch.BoolTensor(tetrahs.shape[0]).to(tetrahs.device)
        ee_mask = ctype < 9
        tv_mask = torch.logical_and(torch.logical_not(ee_mask), ctype < 12)
        vt_mask = ctype >= 12

        intersects[ee_mask] = self.check_intersects_ee(tetrahs[ee_mask])
        intersects[tv_mask] = self.check_intersects_tv(tetrahs[tv_mask, :, :3].permute(0, 2, 1), tetrahs[tv_mask, :, 3])
        intersects[vt_mask] = self.check_intersects_tv(tetrahs[vt_mask, :, 1:].permute(0, 2, 1), tetrahs[vt_mask, :, 0])

        return intersects

    def make_tetrahedrons(self, triangles, deltas, minimal_collisions):
        idx0 = minimal_collisions[:, 0]
        idx1 = minimal_collisions[:, 1]
        ctype = minimal_collisions[:, 2]

        triangles_0 = triangles[idx0]
        triangles_1 = triangles[idx1]
        deltas_0 = deltas[idx0]
        deltas_1 = deltas[idx1]

        triangles_cat = torch.cat([triangles_0, triangles_1], dim=1)
        deltas_cat = torch.cat([deltas_0, deltas_1], dim=1)
        cat_idxs = self.type2ind[ctype].unsqueeze(-1).repeat(1, 1, 3)

        tetrahs = torch.gather(triangles_cat, 1, cat_idxs).permute(0, 2, 1)
        tetrah_ds = torch.gather(deltas_cat, 1, cat_idxs).permute(0, 2, 1)

        matrix = tetrahs[:, :, :3] - tetrahs[:, :, 3:]
        matrix_d = tetrah_ds[:, :, :3] - tetrah_ds[:, :, 3:]

        coef3 = self.coef3(matrix, matrix_d)
        coef2 = self.coef2(matrix, matrix_d)
        coef1 = self.coef1(matrix, matrix_d)
        coef0 = self.coef0(matrix, matrix_d)

        roots = solve_torch(coef3, coef2, coef1, coef0)
        roots_mask = (roots >= 0) * (roots <= 1)

        tetrahs = tetrahs[roots_mask]
        tetrah_ds = tetrah_ds[roots_mask]
        roots = roots[roots_mask][:, None, None]
        ctype = ctype[roots_mask]
        tri_pairs = minimal_collisions[roots_mask, :2]

        tetrahs_opt = tetrahs + tetrah_ds * roots

        intersects_mask = self.check_intersects(tetrahs_opt, ctype)

        roots = roots[intersects_mask]
        ctype = ctype[intersects_mask]
        tri_pairs = tri_pairs[intersects_mask]

        return tri_pairs, ctype, roots


def get_static_proximity(vertices, faces, radius, n_candidates_per_triangle=32):
    """

    :param vertices: torch.FloatTensor Vx3
    :param faces: torch.LongTensor Fx3
    :param radius: float
    :return: torch.LongTensor Cx3
    """
    triangles = vertices[faces].unsqueeze(dim=0).contiguous()
    bboxes, tree = collisions.bvh(triangles)
    proximity_tensor, = collisions.find_proximity(bboxes, tree, triangles, radius, n_candidates_per_triangle)
    proximity_tensor = proximity_tensor[0]
    mask = proximity_tensor[:, 0] >= 0
    proximity_tensor = proximity_tensor[mask, :]
    return proximity_tensor


def get_continuous_collisions(vertices_from, vertices_to, faces, n_candidates_per_triangle=32,
                              n_collisions_per_triangle=16):
    """

    :param vertices: torch.FloatTensor Vx3
    :param faces: torch.LongTensor Fx3
    :param radius: float
    :return: cont_collisions: torch.LongTensor Cx3, roots: torch.FloatTensor Cx1
    """
    # print(n_candidates_per_triangle, n_collisions_per_triangle)
    triangles_from = vertices_from[faces].unsqueeze(dim=0).contiguous()
    triangles_to = vertices_to[faces].unsqueeze(dim=0).contiguous()

    bboxes, tree = collisions.bvh_motion(triangles_from, triangles_to)
    cont_collisions, roots = collisions.find_collisions_continuous(bboxes, tree, triangles_from, triangles_to,
                                                                   n_candidates_per_triangle, n_collisions_per_triangle)
    cont_collisions = cont_collisions[0]
    roots = roots[0]
    # print(roots.shape)
    mask = cont_collisions[:, 0] >= 0
    cont_collisions = cont_collisions[mask, :]
    roots = roots[mask, :]

    return cont_collisions, roots


class CollisionHelper:
    def __init__(self, device):
        self.type2ind = make_type2ind().to(device)

    # @staticmethod
    # def preprocess_collision_tensor(collision_tensor):
    #     print('****************************')
    #     ctype = collision_tensor[:, 2]
    #     vertex_triangle_mask = ctype >= 12
    #     print('vertex_triangle_mask', vertex_triangle_mask)
    #     if vertex_triangle_mask.sum().sum() == 0:
    #         return collision_tensor
    #
    #     collision_tensor[vertex_triangle_mask][:, [0, 1]] = collision_tensor[vertex_triangle_mask][:, [1, 0]]
    #     collision_tensor[vertex_triangle_mask, 2] -= 3
    #
    #     return collision_tensor

    def _choose_pairs_common(self, vertex_inds, idx_pairs, distances, pairs_per_collision, choose):
        pairs = vertex_inds[:, idx_pairs]
        from_idxs = torch.arange(pairs.shape[0]).to(pairs.device).unsqueeze(1).repeat(1, pairs.shape[1])

        if pairs_per_collision >= idx_pairs.shape[0]:
            pairs = pairs.reshape(-1, 2)
            from_idxs = from_idxs.reshape(-1)
            return pairs, from_idxs
        if choose in ['closest', 'farthest']:
            argsort = torch.argsort(distances, 1, choose == 'farthest')
            argsort = argsort[:, :pairs_per_collision]

            pairs_chosen = torch.gather(pairs, 1, argsort.unsqueeze(-1).repeat(1, 1, 2)).reshape(-1, 2)
            from_idxs_chosen = torch.gather(from_idxs, 1, argsort).reshape(-1)
            return pairs_chosen, from_idxs_chosen
        if choose == 'random':
            rand_ids = torch.multinomial(torch.ones_like(distances), pairs_per_collision)
            pairs_chosen = torch.gather(pairs, 1, rand_ids.unsqueeze(-1).repeat(1, 1, 2)).reshape(-1, 2)
            from_idxs_chosen = torch.gather(from_idxs, 1, rand_ids).reshape(-1)
            return pairs_chosen, from_idxs_chosen
        raise Exception("argument `choose` should be one of ['closest', 'farthest', 'random']")

    def choose_pairs_edge_edge(self, tetrahs, vertex_inds, pairs_per_collision, choose='closest'):

        points1 = tetrahs[:, :2]
        points2 = tetrahs[:, 2:]
        distances = (points1.unsqueeze(2) - points2.unsqueeze(1)).pow(2).sum(dim=-1).sqrt()
        distances = distances.reshape(distances.shape[0], -1)

        idx_pairs = torch.cartesian_prod(torch.arange(2), torch.arange(2, 4))
        pairs_chosen, from_idxs_chosen = self._choose_pairs_common(vertex_inds, idx_pairs, distances,
                                                                   pairs_per_collision, choose)
        return pairs_chosen, from_idxs_chosen

    def choose_pairs_triangle_vertex(self, tetrahs, vertex_inds, pairs_per_collision, choose='closest'):
        points1 = tetrahs[:, :3]
        points2 = tetrahs[:, 3:]
        distances = (points1.unsqueeze(2) - points2.unsqueeze(1)).pow(2).sum(dim=-1).sqrt()
        distances = distances.reshape(distances.shape[0], -1)

        idx_pairs = torch.cartesian_prod(torch.arange(3), torch.arange(3, 4))
        pairs_chosen, from_idxs_chosen = self._choose_pairs_common(vertex_inds, idx_pairs, distances,
                                                                   pairs_per_collision, choose)
        return pairs_chosen, from_idxs_chosen

    def print_collisions_as_vertex_ids(self, collisions, faces):
        collisions = self.preprocess_collision_tensor(collisions.clone())


        # collisions = collisions[:0]
        idx0 = collisions[:, 0]
        idx1 = collisions[:, 1]
        ctype = collisions[:, 2]

        faces1 = faces[idx0]
        faces2 = faces[idx1]
        faces_cat = torch.cat([faces1, faces2], dim=1)

        cat_idxs = self.type2ind[ctype]
        vertex_inds = torch.gather(faces_cat, 1, cat_idxs)

        ee_mask = ctype <= 8
        vt_mask = ctype > 8
        vertex_inds_ee = vertex_inds[ee_mask]
        vertex_inds_vt = vertex_inds[vt_mask]

        print("Vertex-Triangle Stencils")
        for i in range(vertex_inds_vt.shape[0]):
            q0, q1, q2, p = vertex_inds_vt[i]
            print(f"({p})<->({q0, q1, q2})")

        print("Edge-Edge Stencils")
        for i in range(vertex_inds_ee.shape[0]):
            p0, p1, q0, q1 = vertex_inds_ee[i]
            print(f"({p0}, {p1})<->({q0}, {q1})")

    def choose_pairs(self, triangles_cat, faces_cat, ctype, pairs_per_collision, choose):
        ee_mask = ctype < 9
        tv_mask = ctype >= 9

        cat_idxs = self.type2ind[ctype]
        tetrahs = torch.gather(triangles_cat, 1, cat_idxs.unsqueeze(-1).repeat(1, 1, 3))
        vertex_inds = torch.gather(faces_cat, 1, cat_idxs)

        coll_idxs_all = torch.arange(vertex_inds.shape[0]).to(vertex_inds.device)

        all_pairs_list = []
        from_ids_list = []

        if ee_mask.sum() > 0:
            pairs_edge_edge, from_ids_ee = self.choose_pairs_edge_edge(tetrahs[ee_mask], vertex_inds[ee_mask],
                                                                       pairs_per_collision, choose)
            from_ids_ee_adj = coll_idxs_all[ee_mask][from_ids_ee]
            all_pairs_list.append(pairs_edge_edge)
            from_ids_list.append(from_ids_ee_adj)

        if tv_mask.sum() > 0:
            pairs_triangle_vertex, from_ids_tv = self.choose_pairs_triangle_vertex(tetrahs[tv_mask],
                                                                                   vertex_inds[tv_mask],
                                                                                   pairs_per_collision, choose)
            from_ids_tv_adj = coll_idxs_all[from_ids_tv]
            all_pairs_list.append(pairs_triangle_vertex)
            from_ids_list.append(from_ids_tv_adj)

        if len(all_pairs_list) == 0:
            all_pairs = torch.zeros(0, 2).long().to(triangles_cat.device)
            from_ids_all = torch.zeros(0, 1).long().to(triangles_cat.device)
            return all_pairs, from_ids_all

        all_pairs = torch.cat(all_pairs_list)
        from_ids_all = torch.cat(from_ids_list)

        return all_pairs, from_ids_all

    def filter_unique(self, pairs):
        if pairs.shape[0] == 0:
            return pairs
        argsort = torch.argsort(pairs, dim=1)
        pairs = torch.gather(pairs, 1, argsort)
        pairs = torch.unique(pairs, dim=0)
        return pairs

    def filter_unique_minimal_roots(self, pairs, roots):
        if pairs.shape[0] == 0:
            return pairs, roots
        argsort = torch.argsort(pairs, dim=1)
        pairs = torch.gather(pairs, 1, argsort)
        pairs, inverse = torch.unique(pairs, dim=0, return_inverse=True)
        roots, _ = scatter_min(roots[:, 0], inverse)
        return pairs, roots

    def contcollisions2nodepairs(self, collisions, roots, vertices_from, vertices_to, faces, pairs_per_collision=4,
                                 choose='closest'):
        """
        :param collisions: torch.LongTensor Cx3
        :param roots: torch.LongTensor Cx1
        :param vertices_from: torch.FloatTensor Vx3
        :param vertices_to: torch.FloatTensor Vx3
        :param faces: torch.LongTensor Fx3
        :param pairs_per_collision: int [1,4]
        :param choose: str ['closest', 'farthest', 'random']
        :return: all_pairs: torch.LongTensor Px2, all_roots: torch.FloatTensor Px1
        """
        collisions = self.preprocess_collision_tensor(collisions.clone())
        # collisions = collisions[:0]
        idx0 = collisions[:, 0]
        idx1 = collisions[:, 1]
        ctype = collisions[:, 2]

        faces1 = faces[idx0]
        faces2 = faces[idx1]
        faces_cat = torch.cat([faces1, faces2], dim=1)

        triangles_cat_from = vertices_from[faces_cat]
        triangles_cat_to = vertices_to[faces_cat]
        triangles_cat = triangles_cat_from + (triangles_cat_to - triangles_cat_from) * roots[..., None]

        all_pairs, from_ids = self.choose_pairs(triangles_cat, faces_cat, ctype, pairs_per_collision, choose)
        all_roots = roots[from_ids]

        if all_pairs.shape[0] == 0:
            return all_pairs, all_roots[:, 0]
        all_pairs, all_roots = self.filter_unique_minimal_roots(all_pairs, all_roots)

        return all_pairs, all_roots[:, None]

    def collisions2nodepairs(self, collisions, vertices, faces, pairs_per_collision=4, choose='closest'):
        """

        :param collisions: torch.LongTensor Cx3
        :param vertices: torch.FloatTensor Vx3
        :param faces: torch.LongTensor Fx3
        :param pairs_per_collision: int [1,4]
        :param choose: str ['closest', 'farthest', 'random']
        :return: torch.LongTensor Px2
        """
        collisions = self.preprocess_collision_tensor(collisions.clone())

        idx0 = collisions[:, 0]
        idx1 = collisions[:, 1]
        ctype = collisions[:, 2]

        faces1 = faces[idx0]
        faces2 = faces[idx1]

        faces_cat = torch.cat([faces1, faces2], dim=1)
        triangles_cat = vertices[faces_cat]
        all_pairs, _ = self.choose_pairs(triangles_cat, faces_cat, ctype, pairs_per_collision, choose)
        all_pairs = self.filter_unique(all_pairs)

        return all_pairs

    def make_rigid_impact_zones(self, faces, collisions, edges=None, nonpenetrating_mask=None):
        collisions = self.preprocess_collision_tensor(collisions.clone())
        idx0 = collisions[:, 0]
        idx1 = collisions[:, 1]
        ctype = collisions[:, 2]

        faces1 = faces[idx0]
        faces2 = faces[idx1]
        faces_cat = torch.cat([faces1, faces2], dim=1)

        if nonpenetrating_mask is not None:
            nonpenetrating_mask = nonpenetrating_mask[:, 0]

        cat_idxs = self.type2ind[ctype]
        vertex_inds = torch.gather(faces_cat, 1, cat_idxs)

        edges_list = []
        if edges is not None:
            edges_list.append(edges)

        for i in range(4):
            from_ids = vertex_inds[:, i]
            to_ids = vertex_inds[:, (i + 1) % 4]
            es = torch.stack([from_ids, to_ids], dim=1)
            if nonpenetrating_mask is not None:
                es_pm = nonpenetrating_mask[es].any(dim=1)
                es = es[es_pm]

            es = es.cpu().numpy()
            edges_list.append(es)

        edges = np.concatenate(edges_list, axis=0)
        edges = np.concatenate(edges_list, axis=0)
        edges = torch.tensor(edges)
        edges = self.filter_unique(edges)
        edges = edges.numpy()

        G = nx.Graph()
        G.add_edges_from(edges)
        components = [torch.LongTensor(list(x)).to(faces.device) for x in nx.connected_components(G)]
        return components, edges

    def make_rigid_impact_zones_tritri(self, faces, collisions_tri, edges=None):
        faces_tri = faces[collisions_tri]
        vertex_inds = faces_tri.reshape(faces_tri.shape[0], -1)

        edges_list = []
        if edges is not None:
            edges_list.append(edges)

        for i in range(6):
            from_ids = vertex_inds[:, i]
            to_ids = vertex_inds[:, (i + 1) % 6]
            es = torch.stack([from_ids, to_ids], dim=1).cpu().numpy()
            edges_list.append(es)

        edges = np.concatenate(edges_list, axis=0)
        edges = torch.tensor(edges)
        edges = self.filter_unique(edges)
        edges = edges.numpy()

        G = nx.Graph()
        G.add_edges_from(edges)
        components = [torch.LongTensor(list(x)).to(faces.device) for x in nx.connected_components(G)]
        return components, edges

    @staticmethod
    def preprocess_collision_tensor(collision_tensor):
        ctype = collision_tensor[:, 2]
        vertex_triangle_mask = ctype >= 12

        # print('vertex_triangle_mask', vertex_triangle_mask)
        if vertex_triangle_mask.sum().sum() == 0:
            return collision_tensor

        # print('AAAAAAAA', collision_tensor[vertex_triangle_mask][:, :2])
        # print('BBBBBBBBB', collision_tensor[vertex_triangle_mask][:, [1, 0]])
        collision_tensor_upd = collision_tensor[vertex_triangle_mask][:, [1, 0]]
        collision_tensor_upd = torch.cat([collision_tensor_upd, collision_tensor[vertex_triangle_mask][:, 2:]], dim=1)
        # print('collision_tensor_upd', collision_tensor_upd)

        collision_tensor[vertex_triangle_mask] = collision_tensor_upd
        # collision_tensor[vertex_triangle_mask][:, :2] = collision_tensor[vertex_triangle_mask][:, [1, 0]]
        collision_tensor[vertex_triangle_mask, 2] -= 3
        # print('CCCCCCc', collision_tensor[vertex_triangle_mask][:, ])


        # print('collision_tensor NEW', collision_tensor)

        return collision_tensor

    def _choose_pairs_common(self, vertex_inds, idx_pairs, distances, pairs_per_collision, choose):
        pairs = vertex_inds[:, idx_pairs]
        from_idxs = torch.arange(pairs.shape[0]).to(pairs.device).unsqueeze(1).repeat(1, pairs.shape[1])
        if pairs_per_collision >= idx_pairs.shape[0]:
            pairs = pairs.reshape(-1, 2)
            from_idxs = from_idxs.reshape(-1)
            return pairs, from_idxs
        if choose in ['closest', 'farthest']:
            argsort = torch.argsort(distances, 1, choose == 'farthest')
            argsort = argsort[:, :pairs_per_collision]
            pairs_chosen = torch.gather(pairs, 1, argsort.unsqueeze(-1).repeat(1, 1, 2)).reshape(-1, 2)
            from_idxs_chosen = torch.gather(from_idxs, 1, argsort).reshape(-1)
            return pairs_chosen, from_idxs_chosen
        if choose == 'random':
            probs = torch.ones_like(distances)
            rand_ids = torch.multinomial(probs, pairs_per_collision)
            pairs_chosen = torch.gather(pairs, 1, rand_ids.unsqueeze(-1).repeat(1, 1, 2)).reshape(-1, 2)
            from_idxs_chosen = torch.gather(from_idxs, 1, rand_ids).reshape(-1)
            return pairs_chosen, from_idxs_chosen
        raise Exception("argument `choose` should be one of ['closest', 'farthest', 'random']")

    def choose_pairs_edge_edge(self, tetrahs, vertex_inds, pairs_per_collision, choose='closest'):

        points1 = tetrahs[:, :2]
        points2 = tetrahs[:, 2:]
        distances = (points1.unsqueeze(2) - points2.unsqueeze(1)).pow(2).sum(dim=-1).sqrt()
        distances = distances.reshape(distances.shape[0], -1)

        idx_pairs = torch.cartesian_prod(torch.arange(2), torch.arange(2, 4))
        pairs_chosen, from_idxs_chosen = self._choose_pairs_common(vertex_inds, idx_pairs, distances,
                                                                   pairs_per_collision, choose)
        return pairs_chosen, from_idxs_chosen

    def choose_pairs_triangle_vertex(self, tetrahs, vertex_inds, pairs_per_collision, choose='closest'):
        points1 = tetrahs[:, :3]
        points2 = tetrahs[:, 3:]
        distances = (points1.unsqueeze(2) - points2.unsqueeze(1)).pow(2).sum(dim=-1).sqrt()
        distances = distances.reshape(distances.shape[0], -1)

        idx_pairs = torch.cartesian_prod(torch.arange(3), torch.arange(3, 4))
        pairs_chosen, from_idxs_chosen = self._choose_pairs_common(vertex_inds, idx_pairs, distances,
                                                                   pairs_per_collision, choose)
        return pairs_chosen, from_idxs_chosen

    def choose_pairs(self, triangles_cat, faces_cat, ctype, pairs_per_collision, choose):
        ee_mask = ctype < 9
        tv_mask = ctype >= 9

        cat_idxs = self.type2ind[ctype]
        tetrahs = torch.gather(triangles_cat, 1, cat_idxs.unsqueeze(-1).repeat(1, 1, 3))
        vertex_inds = torch.gather(faces_cat, 1, cat_idxs)

        coll_idxs_all = torch.arange(vertex_inds.shape[0]).to(vertex_inds.device)

        all_pairs_list = []
        from_ids_list = []

        if ee_mask.sum() > 0:
            pairs_edge_edge, from_ids_ee = self.choose_pairs_edge_edge(tetrahs[ee_mask], vertex_inds[ee_mask],
                                                                       pairs_per_collision, choose)
            from_ids_ee_adj = coll_idxs_all[ee_mask][from_ids_ee]
            all_pairs_list.append(pairs_edge_edge)
            from_ids_list.append(from_ids_ee_adj)

        if tv_mask.sum() > 0:
            pairs_triangle_vertex, from_ids_tv = self.choose_pairs_triangle_vertex(tetrahs[tv_mask],
                                                                                   vertex_inds[tv_mask],
                                                                                   pairs_per_collision, choose)
            from_ids_tv_adj = coll_idxs_all[from_ids_tv]
            all_pairs_list.append(pairs_triangle_vertex)
            from_ids_list.append(from_ids_tv_adj)

        if len(all_pairs_list) == 0:
            all_pairs = torch.zeros(0, 2).long().to(triangles_cat.device)
            from_ids_all = torch.zeros(0, 1).long().to(triangles_cat.device)
            return all_pairs, from_ids_all

        all_pairs = torch.cat(all_pairs_list)
        from_ids_all = torch.cat(from_ids_list)

        mask = (all_pairs >= 0).all(dim=-1)
        all_pairs = all_pairs[mask]
        from_ids_all = from_ids_all[mask]

        return all_pairs, from_ids_all

    def collisions2nodefacepairs(self, collisions, faces, unique=False):
        """

        :param collisions: torch.LongTensor Cx3
        :param faces: torch.LongTensor Fx3
        :return: torch.LongTensor Cx1,  torch.LongTensor Cx1,
        """
        collisions = collisions.clone()

        # print(1111)
        collisions = self.preprocess_collision_tensor(collisions.clone())

        # print('collisions preprocessed', collisions)


        idx0 = collisions[:, 0]
        idx1 = collisions[:, 1]
        ctype = collisions[:, 2]

        trinode_mask = ctype > 8
        faces_to = idx0[trinode_mask]
        faces_from = idx1[trinode_mask]
        ctype = ctype[trinode_mask]

        node_ids = ctype % 3

        faces2 = faces[faces_from]

        nodes = torch.gather(faces2, 1, node_ids[:, None])
        faces_to = faces_to[:, None]
        if unique:
            pairs = torch.cat([nodes, faces_to], dim=1)
            pairs = torch.unique(pairs, dim=0)
            nodes = pairs[:, :1]
            faces_to = pairs[:, 1:]


        return nodes, faces_to



def add_collisions(path, obj_type='cloth', overwrite=False):
    device = 'cuda:0'

    with open(path, 'rb') as f:
        seq_dict = pickle.load(f)

    is_faces_mask = False
    N = len(seq_dict['pred'])

    if obj_type == 'cloth':
        if not overwrite and 'colliding_faces' in seq_dict:
            return

        faces = torch.tensor(seq_dict['cloth_faces'].astype(np.int64),
                             dtype=torch.long,
                             device=device).contiguous()

        if 'faces_mask' in seq_dict:
            faces_mask = torch.tensor(seq_dict['faces_mask'].astype(np.int64),
                                      dtype=torch.long,
                                      device=device).contiguous()
            is_faces_mask = True

        vertices = seq_dict['pred']

    elif obj_type == 'body':
        if not overwrite and 'obstacle_colliding_faces' in seq_dict:
            return

        faces = torch.tensor(seq_dict['obstacle_faces'].astype(np.int64),
                             dtype=torch.long,
                             device=device).contiguous()
        vertices = seq_dict['obstacle']
    elif obj_type == 'both':
        if not overwrite and 'colliding_faces' in seq_dict and 'obstacle_colliding_faces' in seq_dict:
            return

        N_cloth = seq_dict['pred'].shape[1]

        faces_cloth = seq_dict['cloth_faces']
        faces_obstacle = seq_dict['obstacle_faces']
        faces_obstacle = faces_obstacle + N_cloth

        faces = np.concatenate([faces_cloth, faces_obstacle], axis=0)

        vertices = np.concatenate([seq_dict['pred'], seq_dict['obstacle']], axis=1)

        if 'faces_mask' in seq_dict:
            faces_mask_cloth = torch.tensor(seq_dict['faces_mask'].astype(np.int64),
                                            dtype=torch.long,
                                            device=device).contiguous()
            is_faces_mask = True

            faces_mask = torch.ones_like(faces[:, 0])
            faces_mask[:faces_cloth.shape[0]] = faces_mask_cloth
    else:
        raise ValueError('obj_type should be cloth or body or both')

    collisions_list = []
    for i in tqdm(range(N)):

        vertices_curr = torch.tensor(vertices[i],
                                     dtype=torch.float32, device=device).contiguous()

        collisions_tri = find_close_faces(vertices_curr.double(), faces, threshold=0)

        if is_faces_mask:
            # print('collisions_tri', collisions_tri.shape)
            collisions_tri_mask = faces_mask[collisions_tri].all(dim=-1)
            # print('collisions_tri_mask', collisions_tri_mask.shape)
            collisions_tri = collisions_tri[collisions_tri_mask]
            # print('collisions_tri2', collisions_tri.shape)

        colliding_faces = torch.unique(collisions_tri)

        collisions_list.append(colliding_faces.cpu().numpy())

    if obj_type == 'cloth':
        seq_dict['colliding_faces'] = collisions_list
    elif obj_type == 'body':
        seq_dict['obstacle_colliding_faces'] = collisions_list
    elif obj_type == 'both':
        collision_list_cloth = []
        collision_list_obstacle = []
        F_cloth = seq_dict['cloth_faces'].shape[0]

        for cfs in collisions_list:
            cfs_cloth = cfs[cfs < F_cloth]
            cfs_obstacle = cfs[cfs >= F_cloth] - F_cloth
            collision_list_cloth.append(cfs_cloth)
            collision_list_obstacle.append(cfs_obstacle)

        seq_dict['colliding_faces'] = collision_list_cloth
        seq_dict['obstacle_colliding_faces'] = collision_list_obstacle

    print('tritri:', sum(x.shape[0] for x in collisions_list))

    with open(path, 'wb') as f:
        pickle.dump(seq_dict, f)


def find_nearest_neighbors(fcenters_pos, nodes_pos, cloth_faces, radius, K=10, same_object=True, sqrt_distance=False):
    """
    Find nearest vertices within a given radius for each face and filter out face-node pairs where the node is part of the face.

    Args:
        fcenters_pos (torch.Tensor): Positions of centers of faces of a triangle mesh (Fx3)
        nodes_pos (torch.Tensor): Positions of the nodes (Nx3)
        cloth_faces (torch.Tensor): Node indices corresponding to each face
        radius (float): Distance threshold for filtering face-node pairs
    Returns:
        filtered_neighbors (torch.Tensor): Filtered nearest neighbors tensor in shape [Px2]
    """

    K = min(K, nodes_pos.shape[0])
    # Call knn_points to find 10 nearest neighbors for each face
    nn_dists, nn_idx, _ = knn_points(fcenters_pos.unsqueeze(0), nodes_pos.unsqueeze(0), return_nn=True, K=K)
    if sqrt_distance:
        nn_dists = nn_dists.sqrt()

    # print('\nfcenters_pos', fcenters_pos)
    # print('\nnodes_pos', nodes_pos)
    # print('\nnn_idx', nn_idx)
    # print('\nnn_dists', nn_dists)

    # Remove batch dimension
    nn_dists = nn_dists.squeeze(0)
    nn_idx = nn_idx.squeeze(0)

    # Create face indices tensor with the same shape as nn_idx
    face_indices = torch.arange(nn_idx.shape[0], dtype=torch.long).unsqueeze(1).expand(-1, K).to(fcenters_pos.device)

    if same_object:
        # Check if node is part of the corresponding face
        cloth_faces_expanded = cloth_faces[face_indices]  # Shape: (Fx10x3)
        is_part_of_face = (cloth_faces_expanded == nn_idx.unsqueeze(-1)).any(-1)

        # Filter out face-node pairs where the node is part of the face
        face_indices = face_indices[~is_part_of_face]
        nn_idx = nn_idx[~is_part_of_face]
        nn_dists = nn_dists[~is_part_of_face]

    # Filter out face-node pairs where the distance is greater than or equal to the radius


    # print('\nnn_idx', nn_idx)
    # print('\nnn_dists', nn_dists)

    within_radius = nn_dists < radius
    face_indices_filtered = face_indices[within_radius]
    nn_idx_filtered = nn_idx[within_radius]

    # Concatenate face_indices_filtered and nn_idx_filtered to form a (Px2) tensor
    filtered_neighbors = torch.stack((face_indices_filtered, nn_idx_filtered), dim=1)

    return filtered_neighbors


def filter_inside(facenode_pairs, obj1_pos, obj1_faces, obj2_pos, epsilon=1e-5):
    obj1_triangles = obj1_pos[obj1_faces]

    obj1_triangles4bary = obj1_triangles[facenode_pairs[:, 0]]
    obj2_pos4bary = obj2_pos[facenode_pairs[:, 1]]

    barycoords = get_barycoords(obj2_pos4bary, obj1_triangles4bary)
    inside_mask = (barycoords > -epsilon).all(dim=-1)

    facenode_pairs = facenode_pairs[inside_mask]
    return facenode_pairs


def filter_closest_from_each_side(facenode_pairs, face_normals, fcenters_pos, nodes_pos):
    """
    Filters the facenode_pairs to keep one correspondence to the closest node on the positive and negative side of the face.

    Args:
        facenode_pairs (torch.Tensor): Filtered nearest neighbors tensor in shape [Px2]
        face_normals (torch.Tensor): Normal vectors for each face in the mesh [Fx3]
        fcenters_pos (torch.Tensor): Positions of each face center [Fx3]
        nodes_pos (torch.Tensor): Positions of each node [Nx3]
    Returns:
        filtered_pairs (torch.Tensor): Filtered facenode_pairs in shape [Fx2]
    """
    # Get face and node indices from facenode_pairs
    face_indices = facenode_pairs[:, 0]
    node_indices = facenode_pairs[:, 1]

    # Compute the vector from the face center to the node for each pair
    face_to_node_vectors = nodes_pos[node_indices] - fcenters_pos[face_indices]

    # Project the vector onto the normal vector
    projections = (face_to_node_vectors * face_normals[face_indices]).sum(-1)

    # Compute the distances from the face center to the node for each pair
    distances = torch.norm(face_to_node_vectors, dim=-1)

    # Create masks for positive and negative side correspondences
    positive_mask = projections >= 0
    negative_mask = projections < 0

    # Get face indices for positive and negative correspondences
    pos_face_indices = face_indices[positive_mask]
    neg_face_indices = face_indices[negative_mask]

    # Get corresponding distances
    pos_distances = distances[positive_mask]
    neg_distances = distances[negative_mask]

    # Get node indices for positive and negative correspondences
    pos_node_indices = node_indices[positive_mask]
    neg_node_indices = node_indices[negative_mask]

    # Find minimum distances for each face
    _, pos_min_indices = torch.unique(pos_face_indices, return_inverse=True)
    _, neg_min_indices = torch.unique(neg_face_indices, return_inverse=True)

    pos_min_distances = pos_distances.scatter(0, pos_min_indices, pos_distances).gather(0, pos_min_indices)
    neg_min_distances = neg_distances.scatter(0, neg_min_indices, neg_distances).gather(0, neg_min_indices)
    # Create masks for closest nodes
    pos_closest_mask = pos_distances == pos_min_distances
    neg_closest_mask = neg_distances == neg_min_distances

    # Find the closest node indices
    pos_closest_node_indices = pos_node_indices[pos_closest_mask]
    neg_closest_node_indices = neg_node_indices[neg_closest_mask]
    all_closest_node_indices = torch.cat([pos_closest_node_indices, neg_closest_node_indices], dim=0)

    pos_face_indices = pos_face_indices[pos_closest_mask]
    neg_face_indices = neg_face_indices[neg_closest_mask]
    all_face_indices = torch.cat([pos_face_indices, neg_face_indices], dim=0)

    # Combine the closest node indices for positive and negative correspondences
    filtered_pairs = torch.stack((all_face_indices, all_closest_node_indices), dim=1)


    return filtered_pairs


def node_to_node_correspondences(facenode_pairs, nodes_pos, cloth_faces):
    """
    Determines which of the face nodes is closest to the correspondent node and outputs a tensor with node indices.

    Args:
        facenode_pairs (torch.Tensor): Filtered nearest neighbors tensor in shape [Px2]
        nodes_pos (torch.Tensor): Positions of each node [Nx3]
        cloth_faces (torch.Tensor): Indices of the nodes in each face [Fx3]
    Returns:
        node_correspondences (torch.Tensor): Tensor with node indices in shape [Px2]
    """
    # Extract the node indices of the corresponding faces
    face_node_indices = cloth_faces[facenode_pairs[:, 0]]
    node_indices = facenode_pairs[:, 1].unsqueeze(-1).expand(-1, 3)

    # Compute the distance between the corresponding nodes
    distances = torch.norm(nodes_pos[node_indices] - nodes_pos[face_node_indices], dim=-1)

    # Find the minimum distance and its index for each face
    min_distances, min_indices = distances.min(dim=1)

    # Create the output tensor with node indices
    node_correspondences = torch.stack((face_node_indices[torch.arange(face_node_indices.shape[0]), min_indices],
                                        facenode_pairs[:, 1]), dim=1)

    return node_correspondences


def get_all_world_edges(cloth_pos, obstacle_pos, cloth_faces, f_normals_f, bary_epsilon=0.2, radius=3e-2, K=10, one_per_side=True, sqrt_distance=False):
    """
    Computes the world edges for the cloth mesh.

    Args:
        cloth_pos (torch.Tensor): Positions of the cloth mesh [Nx3]
        obstacle_pos (torch.Tensor): Positions of the obstacle mesh [Mx3]
        cloth_faces (torch.Tensor): Indices of the nodes in each face [Fx3]
        f_normals_f (torch.nn.Module): Function that computes the normal vectors for each face
        bary_epsilon (float): Epsilon value for barycentric coordinates
        radius (float): Radius for nearest neighbor search
        K (int): Number of nearest neighbors to search for
        one_per_side (bool): Whether to only keep one edge per side

    Returns:
        world_edges (torch.Tensor): Tensor with world edges in shape [Ex2]
        (Optional) body_edges (torch.Tensor): Tensor with body edges in shape [Bx2]
    """
    cloth_fcenter_pos = gather(cloth_pos, cloth_faces, 0, 1, 1).mean(dim=-2)
    cloth_face_normals = f_normals_f(cloth_pos.unsqueeze(0), cloth_faces.unsqueeze(0))[0]

    if obstacle_pos is None:
        combined_pos = cloth_pos
    else:
        combined_pos = torch.cat([cloth_pos, obstacle_pos], dim=0)


    face2node_pairs = find_nearest_neighbors(cloth_fcenter_pos, combined_pos, cloth_faces, radius=radius, K=K, sqrt_distance=sqrt_distance)
    # print('\nface2node_pairs 1 ', face2node_pairs)

    face2node_pairs = filter_inside(face2node_pairs, combined_pos, cloth_faces, combined_pos, epsilon=bary_epsilon)

    # print('\nface2node_pairs 2 ', face2node_pairs)
    if one_per_side:
        face2node_pairs = filter_closest_from_each_side(face2node_pairs, cloth_face_normals, cloth_fcenter_pos, combined_pos)

    # print('face2node_pairs', face2node_pairs.shape)
    node2node_pairs = node_to_node_correspondences(face2node_pairs, combined_pos, cloth_faces)


    if obstacle_pos is not None:
        N_cloth = cloth_pos.shape[0]
        cc_mask = node2node_pairs[:, 1] < N_cloth
        cb_mask = node2node_pairs[:, 1] >= N_cloth

        world_edges = node2node_pairs[cc_mask]
        body_edges = node2node_pairs[cb_mask]
        body_edges[:, 1] -= N_cloth
        return world_edges, body_edges
    else:
        return node2node_pairs, None

def find_nearest_neighbors_debug(fcenters_pos, nodes_pos, cloth_faces, radius, K=10, same_object=True):
    """
    Find nearest vertices within a given radius for each face and filter out face-node pairs where the node is part of the face.

    Args:
        fcenters_pos (torch.Tensor): Positions of centers of faces of a triangle mesh (Fx3)
        nodes_pos (torch.Tensor): Positions of the nodes (Nx3)
        cloth_faces (torch.Tensor): Node indices corresponding to each face
        radius (float): Distance threshold for filtering face-node pairs
    Returns:
        filtered_neighbors (torch.Tensor): Filtered nearest neighbors tensor in shape [Px2]
    """


    # filter out nodes_pos whose index is not in cloth_faces
    # nodes_pos = nodes_pos[cloth_faces.flatten()]
    # fcenters_pos = fcenters_pos[1:]
    # nodes_pos = nodes_pos[:1]
    #
    # dd = (fcenters_pos - nodes_pos) ** 2
    # dd = np.sqrt(dd.sum())
    # print('dd', dd)


    # Call knn_points to find 10 nearest neighbors for each face
    nn_dists, nn_idx, _ = knn_points(fcenters_pos.unsqueeze(0), nodes_pos.unsqueeze(0), return_nn=True, K=K)
    nn_dists = nn_dists.sqrt()

    # Remove batch dimension
    nn_dists = nn_dists.squeeze(0)
    nn_idx = nn_idx.squeeze(0)

    # print('fcenters_pos', fcenters_pos)
    # print('nodes_pos', nodes_pos)
    #
    # print('nn_idx', nn_idx)
    # print('nn_dists', nn_dists)


    # Create face indices tensor with the same shape as nn_idx
    face_indices = torch.arange(nn_idx.shape[0], dtype=torch.long).unsqueeze(1).expand(-1, K).to(fcenters_pos.device)

    if same_object:
        # Check if node is part of the corresponding face
        cloth_faces_expanded = cloth_faces[face_indices]  # Shape: (Fx10x3)
        is_part_of_face = (cloth_faces_expanded == nn_idx.unsqueeze(-1)).any(-1)

        # Filter out face-node pairs where the node is part of the face
        face_indices = face_indices[~is_part_of_face]

        nn_idx = nn_idx[~is_part_of_face]
        nn_dists = nn_dists[~is_part_of_face]

    # filter out nn_ids whose values do not occur in cloth_faces
    # mask = torch.any(nn_idx.unsqueeze(-1) == cloth_faces.reshape(-1).unsqueeze(0), dim=-1)
    # nn_idx = nn_idx[mask]
    # nn_dists = nn_dists[mask]
    # face_indices = face_indices[mask]

    # print('face_indices', face_indices)
    # print('nn_dists', nn_dists)
    # print('nn_idx', nn_idx)


    # Filter out face-node pairs where the distance is greater than or equal to the radius
    within_radius = nn_dists < radius
    face_indices_filtered = face_indices[within_radius]

    nn_idx_filtered = nn_idx[within_radius]
    # print('nn_idx_filtered', nn_idx_filtered.shape)

    # Concatenate face_indices_filtered and nn_idx_filtered to form a (Px2) tensor
    filtered_neighbors = torch.stack((face_indices_filtered, nn_idx_filtered), dim=1)

    return filtered_neighbors

def get_all_world_edges_debug(cloth_pos, obstacle_pos, cloth_faces, f_normals_f, bary_epsilon=0.2, radius=3e-2, K=10, one_per_side=True):
    """
    Computes the world edges for the cloth mesh.

    Args:
        cloth_pos (torch.Tensor): Positions of the cloth mesh [Nx3]
        obstacle_pos (torch.Tensor): Positions of the obstacle mesh [Mx3]
        cloth_faces (torch.Tensor): Indices of the nodes in each face [Fx3]
        f_normals_f (torch.nn.Module): Function that computes the normal vectors for each face
        bary_epsilon (float): Epsilon value for barycentric coordinates
        radius (float): Radius for nearest neighbor search
        K (int): Number of nearest neighbors to search for
        one_per_side (bool): Whether to only keep one edge per side

    Returns:
        world_edges (torch.Tensor): Tensor with world edges in shape [Ex2]
        (Optional) body_edges (torch.Tensor): Tensor with body edges in shape [Bx2]
    """
    cloth_fcenter_pos = gather(cloth_pos, cloth_faces, 0, 1, 1).mean(dim=-2)
    cloth_face_normals = f_normals_f(cloth_pos.unsqueeze(0), cloth_faces.unsqueeze(0))[0]

    if obstacle_pos is None:
        combined_pos = cloth_pos
    else:
        combined_pos = torch.cat([cloth_pos, obstacle_pos], dim=0)


    face2node_pairs = find_nearest_neighbors_debug(cloth_fcenter_pos, combined_pos, cloth_faces, radius=radius, K=K)
    return face2node_pairs, None
    # print(face2node_pairs)
    face2node_pairs = filter_inside(face2node_pairs, combined_pos, cloth_faces, combined_pos, epsilon=bary_epsilon)
    print('face2node_pairs inside', face2node_pairs.shape)
    # print(face2node_pairs)
    if one_per_side:
        face2node_pairs = filter_closest_from_each_side(face2node_pairs, cloth_face_normals, cloth_fcenter_pos, combined_pos)

    # print('face2node_pairs', face2node_pairs.shape)
    node2node_pairs = node_to_node_correspondences(face2node_pairs, combined_pos, cloth_faces)
    # print('node2node_pairs', node2node_pairs.shape)

    if obstacle_pos is not None:
        N_cloth = cloth_pos.shape[0]
        cc_mask = node2node_pairs[:, 1] < N_cloth
        cb_mask = node2node_pairs[:, 1] >= N_cloth

        world_edges = node2node_pairs[cc_mask]
        body_edges = node2node_pairs[cb_mask]
        body_edges[:, 1] -= N_cloth
        return world_edges, body_edges
    else:
        return node2node_pairs, None



