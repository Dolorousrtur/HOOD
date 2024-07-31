import os
import pickle
import numpy as np
import torch
from pytorch3d.io.obj_io import _save as pt3dio_saveobj


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


def save_obj(filename, verts, faces, verts_uvs=None, faces_uvs=None):
    if type(verts) != torch.Tensor:
        verts = torch.FloatTensor(verts)
    if type(faces) != torch.Tensor:
        faces = torch.LongTensor(faces)
    if verts_uvs is not None and type(verts_uvs) != torch.Tensor:
        verts_uvs = torch.FloatTensor(verts_uvs)
    if faces_uvs is not None and type(faces_uvs) != torch.Tensor:
        faces_uvs = torch.LongTensor(faces_uvs)

    verts = verts.cpu().detach()
    faces = faces.cpu().detach()


    save_texture = verts_uvs is not None and faces_uvs is not None

    with open(filename, "w") as f:
        pt3dio_saveobj(f, verts, faces, verts_uvs=verts_uvs, faces_uvs=faces_uvs, save_texture=save_texture)


def pickle_load(file):
    """
    Load a pickle file.
    """
    with open(file, 'rb') as f:
        loadout = pickle.load(f)

    return loadout


def pickle_dump(loadout, file):
    """
    Dump a pickle file. Create the directory if it does not exist.
    """
    os.makedirs(os.path.dirname(str(file)), exist_ok=True)

    with open(file, 'wb') as f:
        pickle.dump(loadout, f)