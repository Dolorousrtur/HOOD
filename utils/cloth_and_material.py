from collections import defaultdict

import einops
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Batch

from utils.common import gather, unsorted_segment_sum, add_field_to_pyg_batch


class Cloth:
    def __init__(self, material, always_overwrite_mass=False):
        """
        Maintains a cache of the properties for each garment
        :param material: Material object containing material parameters
        :param always_overwrite_mass: True if the mass of the vertices should be recomputed every time
        """
        self.cache = defaultdict(dict)
        self.material = material
        self.always_overwrite_mass = always_overwrite_mass

    def make_v_mass(self, v, f, device):
        """
        Conpute mass of each vertex
        :param v: vertex positions [Vx3]
        :param f: faces [Fx3]
        :param device: pytorch device
        :return: vertex mass [Vx1]
        """
        v_mass = get_vertex_mass(v, f, self.material.density).to(device)
        v_mass = v_mass.unsqueeze(-1)
        return v_mass

    def make_f_area(self, v, f, device):
        """
        Compute areas of each face
        :param v: vertex positions [Vx3]
        :param f: faces [Fx3]
        :param device: pytorch device
        :return: face areas [Fx1]
        """
        f_area = torch.FloatTensor(get_face_areas(v, f)).to(device)  # +
        f_area = f_area.unsqueeze(-1)
        return f_area

    def make_connectivity(self, f):
        f_connectivity, f_connectivity_edges = get_face_connectivity_combined(f)
        return f_connectivity, f_connectivity_edges

    def make_Dm_inv(self, v, f):
        """
        Conpute inverse of the deformation gradient matrix (used in stretching energy loss)
        :param v: vertex positions [Vx3]
        :param f: faces [Fx3]
        :return: inverse of the deformation gradient matrix [Fx3x3]
        """
        tri_m = gather_triangles(v.unsqueeze(0), f)[0]

        edges = get_shape_matrix(tri_m)
        edges = edges.permute(0, 2, 1)
        edges_2d = edges_3d_to_2d(edges).permute(0, 2, 1)
        Dm_inv = torch.inverse(edges_2d)
        return Dm_inv

    def set_material(self, material):
        self.material = material

    def set_batch(self, batch, overwrite_pos=False):
        """
        Add cloth properties needed for objective computation to the cvpr batch
        Maintains a cache of the properties for each garment

        :param batch: pytorch geometric batch object
        :param overwrite_pos: True is the resting pose of the garments is augmented (is different on every step)
        :return:
        """
        B = batch.num_graphs
        device = batch['cloth'].pos.device

        new_examples_list = []
        for i in range(B):
            example = batch.get_example(i)
            garment_name = example.garment_name

            # resting pose geometru
            v = example['cloth'].rest_pos

            # faces
            f = example['cloth'].faces_batch.T
            if self.always_overwrite_mass:
                v_mass = self.make_v_mass(v, f, device)

            if garment_name in self.cache:
                f_connectivity = self.cache[garment_name]['f_connectivity']
                f_connectivity_edges = self.cache[garment_name]['f_connectivity_edges']
                if not overwrite_pos:
                    if not self.always_overwrite_mass:
                        v_mass = self.cache[garment_name]['v_mass']
                    f_area = self.cache[garment_name]['f_area']
                    Dm_inv = self.cache[garment_name]['Dm_inv']
                else:
                    if not self.always_overwrite_mass:
                        v_mass = self.make_v_mass(v, f, device)
                    f_area = self.make_f_area(v, f, device)
                    Dm_inv = self.make_Dm_inv(v, f)
            else:
                if not self.always_overwrite_mass:
                    v_mass = self.make_v_mass(v, f, device)
                f_area = self.make_f_area(v, f, device)
                f_connectivity, f_connectivity_edges = self.make_connectivity(f)
                Dm_inv = self.make_Dm_inv(v, f)

                self.cache[garment_name]['v_mass'] = v_mass
                self.cache[garment_name]['f_area'] = f_area
                self.cache[garment_name]['f_connectivity'] = f_connectivity
                self.cache[garment_name]['f_connectivity_edges'] = f_connectivity_edges
                self.cache[garment_name]['Dm_inv'] = Dm_inv

            example['cloth'].v_mass = v_mass  # vertex mass [Vx1]
            example['cloth'].f_area = f_area  # face areas [Fx1]

            # list of face pairs connected with each edge [Ex2]
            example['cloth'].f_connectivity = f_connectivity

            # list of edges connecting corresponding face pairs [Ex3]
            example['cloth'].f_connectivity_edges = f_connectivity_edges

            # inverse of the deformation gradient matrix [Fx3x3]
            example['cloth'].Dm_inv = Dm_inv
            new_examples_list.append(example)

        batch = Batch.from_data_list(new_examples_list)
        return batch


class ClothMatAug(Cloth):
    def __init__(self, material, always_overwrite_mass=False):
        super().__init__(material, True)
        self.always_overwrite_mass = always_overwrite_mass

    def set_material(self, material):
        self.material = material

    def make_v_mass(self, v, f, density, device):

        v_mass = get_vertex_mass(v, f, density).to(device)
        v_mass = v_mass.unsqueeze(-1)
        return v_mass

    def set_batch(self, batch, overwrite_pos=False):

        B = batch.num_graphs
        device = batch['cloth'].pos.device

        new_examples_list = []
        for i in range(B):
            example = batch.get_example(i)
            garment_name = example.garment_name

            v = example['cloth'].rest_pos
            f = example['cloth'].faces_batch.T
            density = self.material.density[i].item()

            if self.always_overwrite_mass:
                v_mass = self.make_v_mass(v, f, density, device)

            if garment_name in self.cache:
                f_connectivity = self.cache[garment_name]['f_connectivity']
                f_connectivity_edges = self.cache[garment_name]['f_connectivity_edges']
                if not overwrite_pos:
                    if not self.always_overwrite_mass:
                        v_mass = self.cache[garment_name]['v_mass']
                    f_area = self.cache[garment_name]['f_area']
                    Dm_inv = self.cache[garment_name]['Dm_inv']
                else:
                    if not self.always_overwrite_mass:
                        v_mass = self.make_v_mass(v, f, density, device)
                    f_area = self.make_f_area(v, f, device)
                    Dm_inv = self.make_Dm_inv(v, f)
            else:
                if not self.always_overwrite_mass:
                    v_mass = self.make_v_mass(v, f, density, device)
                f_area = self.make_f_area(v, f, device)

                f_connectivity, f_connectivity_edges = self.make_connectivity(f)
                Dm_inv = self.make_Dm_inv(v, f)

                self.cache[garment_name]['v_mass'] = v_mass
                self.cache[garment_name]['f_area'] = f_area
                self.cache[garment_name]['f_connectivity'] = f_connectivity
                self.cache[garment_name]['f_connectivity_edges'] = f_connectivity_edges
                self.cache[garment_name]['Dm_inv'] = Dm_inv

            example['cloth'].v_mass = v_mass
            example['cloth'].f_area = f_area
            example['cloth'].f_connectivity = f_connectivity
            example['cloth'].f_connectivity_edges = f_connectivity_edges
            example['cloth'].Dm_inv = Dm_inv

            new_examples_list.append(example)

        batch = Batch.from_data_list(new_examples_list)
        return batch


class Material():
    '''
    This class stores parameters for the StVK material model
    '''

    def __init__(self, density,  # Fabric density (kg / m2)
                 lame_mu,
                 lame_lambda,
                 bending_coeff,
                 bending_multiplier=50.0,
                 gravity=None):
        self.density = density

        self.bending_multiplier = bending_multiplier

        self.bending_coeff = bending_coeff
        self.bending_coeff *= bending_multiplier

        # Lamé coefficients
        self.lame_mu = lame_mu
        self.lame_lambda = lame_lambda

        self.gravity = gravity


def edges_3d_to_2d(edges):
    """
    :param edges: Edges in 3D space (in the world coordinate basis) (E, 2, 3)
    :return: Edges in 2D space (in the intrinsic orthonormal basis) (E, 2, 2)
    """
    # Decompose for readability
    device = edges.device

    edges0 = edges[:, 0]
    edges1 = edges[:, 1]

    # Get orthonormal basis
    basis2d_0 = (edges0 / torch.norm(edges0, dim=-1).unsqueeze(-1))
    n = torch.cross(basis2d_0, edges1, dim=-1)
    basis2d_1 = torch.cross(n, edges0, dim=-1)
    basis2d_1 = basis2d_1 / torch.norm(basis2d_1, dim=-1).unsqueeze(-1)

    # Project original edges into orthonormal basis
    edges2d = torch.zeros((edges.shape[0], edges.shape[1], 2)).to(device=device)
    edges2d[:, 0, 0] = (edges0 * basis2d_0).sum(-1)
    edges2d[:, 0, 1] = (edges0 * basis2d_1).sum(-1)
    edges2d[:, 1, 0] = (edges1 * basis2d_0).sum(-1)
    edges2d[:, 1, 1] = (edges1 * basis2d_1).sum(-1)

    return edges2d


class VertexNormalsPYG(nn.Module):
    """
    Computes vertex normals for a mesh represented as a torch_geometric batch
    """

    def __init__(self):
        super().__init__()

    def forward(self, pyg_data, node_key, pos_key):
        """

        :param pyg_data: torch_geometric.data.HeteroData object
        :param node_key: nome of the node type in HeteroData (e.g. `cloth` or `obstacle`)
        :param pos_key: name of the vertex tensor in the pyg_data[node_key] (e.g. `rest_pos` or `pos`)
        :return: updated HeteroData object
        """

        v = pyg_data[node_key][pos_key]  # V x 3
        f = pyg_data[node_key].faces_batch.T  # F x 3
        triangles = gather(v, f, 0, 1, 1)  # F x 3 x 3
        v0, v1, v2 = torch.unbind(triangles, dim=-2)  # F x 3
        e0 = v1 - v0  # F x 3
        e1 = v2 - v1
        e2 = v0 - v2

        # F x 3
        face_normals = torch.linalg.cross(e0, e1) + torch.linalg.cross(e1, e2) + torch.linalg.cross(e2, e0)

        # V x 3
        vn = unsorted_segment_sum(face_normals, f, 0, 1, 1, n_verts=v.shape[0])

        vn = F.normalize(vn, dim=-1)
        pyg_data = add_field_to_pyg_batch(pyg_data, 'normals', vn, node_key, 'pos')
        return pyg_data


class FaceNormals(nn.Module):
    """
    torch Module that computes face normals for a batch of meshes
    """

    def __init__(self, normalize=True):
        """
        :param normalize: Whether to normalize the face normals
        """

        super().__init__()
        self.normalize = normalize

    def forward(self, vertices, faces):
        """

        :param vertices: FloatTensor of shape (batch_size, num_vertices, 3)
        :param faces: LongTensor of shape (batch_size, num_faces, 3)
        :return: face_normals: FloatTensor of shape (batch_size, num_faces, 3)
        """
        v = vertices
        f = faces

        if v.shape[0] > 1 and f.shape[0] == 1:
            f = f.repeat(v.shape[0], 1, 1)

        v_repeat = einops.repeat(v, 'b m n -> b m k n', k=f.shape[-1])
        f_repeat = einops.repeat(f, 'b m n -> b m n k', k=v.shape[-1])
        triangles = torch.gather(v_repeat, 1, f_repeat)

        # Compute face normals
        v0, v1, v2 = torch.unbind(triangles, dim=-2)
        e1 = v0 - v1
        e2 = v2 - v1
        face_normals = torch.linalg.cross(e2, e1)

        if self.normalize:
            face_normals = F.normalize(face_normals, dim=-1)

        return face_normals


def gather_triangles(vertices, faces):
    """
    Generate a tensor of triangles from a tensor of vertices and faces

    :param vertices: FloatTensor of shape (batch_size, num_vertices, 3)
    :param faces: LongTensor of shape (num_faces, 3)
    :return: triangles: FloatTensor of shape (batch_size, num_faces, 3, 3)
    """
    F = faces.shape[-1]
    B, V, C = vertices.shape

    vertices = einops.repeat(vertices, 'b m n -> b m k n', k=F)
    faces = einops.repeat(faces, 'm n -> b m n k', k=C, b=B)
    triangles = torch.gather(vertices, 1, faces)

    return triangles


def get_shape_matrix(x):
    if len(x.shape) == 3:
        return torch.stack([x[:, 0] - x[:, 2], x[:, 1] - x[:, 2]], dim=-1)

    elif len(x.shape) == 4:
        return torch.stack([x[:, :, 0] - x[:, :, 2], x[:, :, 1] - x[:, :, 2]], dim=-1)

    raise NotImplementedError


def get_vertex_connectivity(faces):
    '''
    Returns a list of unique edges in the mesh.
    Each edge contains the indices of the vertices it connects
    '''
    device = 'cpu'
    if type(faces) == torch.Tensor:
        device = faces.device
        faces = faces.detach().cpu().numpy()

    edges = set()
    for f in faces:
        num_vertices = len(f)
        for i in range(num_vertices):
            j = (i + 1) % num_vertices
            edges.add(tuple(sorted([f[i], f[j]])))

    edges = torch.LongTensor(list(edges)).to(device)
    return edges


def get_face_connectivity_combined(faces):
    """
    Finds the faces that are connected in a mesh
    :param faces: LongTensor of shape (num_faces, 3)
    :return: adjacent_faces: pairs of face indices LongTensor of shape (num_edges, 2)
    :return: adjacent_face_edges: pairs of node indices that comprise the edges connecting the corresponding faces
     LongTensor of shape (num_edges, 2)
    """

    device = 'cpu'
    if type(faces) == torch.Tensor:
        device = faces.device
        faces = faces.detach().cpu().numpy()

    edges = get_vertex_connectivity(faces).cpu().numpy()

    G = {tuple(e): [] for e in edges}
    for i, f in enumerate(faces):
        n = len(f)
        for j in range(n):
            k = (j + 1) % n
            e = tuple(sorted([f[j], f[k]]))
            G[e] += [i]

    adjacent_faces = []
    adjacent_face_edges = []

    for key in G:
        if len(G[key]) >= 3:
            G[key] = G[key][:2]
        if len(G[key]) == 2:
            adjacent_faces += [G[key]]
            adjacent_face_edges += [list(key)]

    adjacent_faces = torch.LongTensor(adjacent_faces).to(device)
    adjacent_face_edges = torch.LongTensor(adjacent_face_edges).to(device)

    return adjacent_faces, adjacent_face_edges


def get_vertex_mass(vertices, faces, density):
    '''
    Computes the mass of each vertex according to triangle areas and fabric density
    '''

    vertices = vertices.cpu()
    faces = faces.cpu()

    areas = get_face_areas(vertices, faces)
    triangle_masses = density * areas

    vertex_masses = np.zeros(vertices.shape[0])
    np.add.at(vertex_masses, faces[:, 0], triangle_masses / 3)
    np.add.at(vertex_masses, faces[:, 1], triangle_masses / 3)
    np.add.at(vertex_masses, faces[:, 2], triangle_masses / 3)

    vertex_masses = torch.FloatTensor(vertex_masses)

    return vertex_masses


def get_face_areas(vertices, faces):
    """
    Computes the area of each face in the mesh

    :param vertices: FloatTensor or numpy array of shape (num_vertices, 3)
    :param faces: LongTensor or numpy array of shape (num_faces, 3)
    :return: areas: FloatTensor or numpy array of shape (num_faces,)
    """
    if type(vertices) == torch.Tensor:
        vertices = vertices.detach().cpu().numpy()

    if type(faces) == torch.Tensor:
        faces = faces.detach().cpu().numpy()
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    u = v2 - v0
    v = v1 - v0

    if u.shape[-1] == 2:
        out = np.abs(np.cross(u, v)) / 2.0
    else:
        out = np.linalg.norm(np.cross(u, v), axis=-1) / 2.0
    return out


def load_obj(filename, tex_coords=False):
    """
    Load a mesh from an obj file

    :param filename: path to the obj file
    :param tex_coords: whether to load texture (UV) coordinates
    :return: vertices: numpy array of shape (num_vertices, 3)
    :return: faces: numpy array of shape (num_faces, 3)
    """
    vertices = []
    faces = []
    uvs = []
    faces_uv = []

    with open(filename, 'r') as fp:
        for line in fp:
            line_split = line.split()

            if not line_split:
                continue

            elif tex_coords and line_split[0] == 'vt':
                uvs.append([line_split[1], line_split[2]])

            elif line_split[0] == 'v':
                vertices.append([line_split[1], line_split[2], line_split[3]])

            elif line_split[0] == 'f':
                vertex_indices = [s.split("/")[0] for s in line_split[1:]]
                faces.append(vertex_indices)

                if tex_coords:
                    uv_indices = [s.split("/")[1] for s in line_split[1:]]
                    faces_uv.append(uv_indices)

    vertices = np.array(vertices, dtype=np.float32)
    faces = np.array(faces, dtype=np.int32) - 1

    if tex_coords:
        uvs = np.array(uvs, dtype=np.float32)
        faces_uv = np.array(faces_uv, dtype=np.int32) - 1
        return vertices, faces, uvs, faces_uv

    return vertices, faces


class VertexNormals(nn.Module):
    def __init__(self, normalize=True):
        super().__init__()
        self.normalize = normalize

    def forward(self, vertices, faces):
        v = vertices
        f = faces

        triangles = gather(v, f, 1, 2, 2)

        # Compute face normals
        v0, v1, v2 = torch.unbind(triangles, dim=-2)
        e0 = v1 - v0
        e1 = v2 - v1
        e2 = v0 - v2
        face_normals = torch.linalg.cross(e0, e1) + torch.linalg.cross(e1, e2) + torch.linalg.cross(e2, e0)  # F x 3

        vn = unsorted_segment_sum(face_normals, f, 1, 2, 2)

        vn = F.normalize(vn, dim=-1)
        return vn