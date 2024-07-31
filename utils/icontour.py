from collections import defaultdict
from time import time
import cudf
import cugraph
import numpy as np
import torch
import torch_collisions
import torch_scatter
from utils.io import pickle_dump, save_obj
from utils.common import triangles_to_edges
import networkx as nx


def make_icontour_masks(points_from, faces_to, node2contour, face2contour, enclosed_nodes_mask, icontour_grad_from):
    mask_is_contour = torch.zeros_like(points_from).bool()
    mask_omit = torch.zeros_like(points_from).bool()

    correspondence_enclosed = enclosed_nodes_mask[points_from]

    node2contour_corr = node2contour[points_from]
    face2contour_corr = face2contour[faces_to]

    matching_nodes = ((node2contour_corr == face2contour_corr) & node2contour_corr).any(-1)
    mask_omit[correspondence_enclosed & ~matching_nodes] = True

    icontour_grad_mask = icontour_grad_from.norm(dim=-1) > 0
    mask_is_contour[icontour_grad_mask] = True

    return mask_omit, mask_is_contour



def get_colliding_primitives(collision_tensor, triangles):
    face_id_pairs = collision_tensor[:, :2]
    collision_types = collision_tensor[:, 2]

    type2tid, type2edge = make_type2ind_triedge(triangles.device)

    type2tid_id = type2tid[collision_types].unsqueeze(-1) 
    type2edge = type2edge[collision_types].unsqueeze(-1).repeat(1, 1, 3)
    face_ids_colliding_edges = torch.gather(face_id_pairs, 1, type2tid_id)
    face_ids_colliding_triangles = torch.gather(face_id_pairs, 1, 1 - type2tid_id)

    triangles_colliding_edges = triangles[face_ids_colliding_edges][:, 0]
    colliding_triangles = triangles[face_ids_colliding_triangles][:, 0]

    colliding_edges = torch.gather(triangles_colliding_edges, 1, type2edge)

    return colliding_edges, triangles_colliding_edges, colliding_triangles

def get_tri_edge_intersection_edges(faces, face_id_pairs, collision_types):
    traingles = faces
    triangles_pairs = traingles[face_id_pairs]
    collision_types = collision_types[:, 0]

    type2tid, type2edge = make_type2ind_triedge(faces.device)
    type2tid = type2tid[collision_types].unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 3)
    type2edge = type2edge[collision_types]  

    triangles = torch.gather(triangles_pairs, 1, type2tid).squeeze(1)

    edges = torch.gather(triangles, 1, type2edge)

    return edges

def make_multipass_mask(intersection_edges, face_ids_gathered):
    intersection_triples = torch.cat([intersection_edges, face_ids_gathered[:, None]], -1)
    triplets_unique, inverse_ids, counts = torch.unique(intersection_triples, return_counts=True, return_inverse=True,
                                                        dim=0)
    multipass_mask = counts > 1
    multipass_mask_all = multipass_mask[inverse_ids]
    return multipass_mask_all


def make_type2ind_triedge(device='cuda:0'):
    type2tid = []
    type2edge = []

    type2tid.append(0)
    type2edge.append([0, 1])

    type2tid.append(0)
    type2edge.append([1, 2])

    type2tid.append(0)
    type2edge.append([2, 0])

    type2tid.append(1)
    type2edge.append([0, 1])

    type2tid.append(1)
    type2edge.append([1, 2])

    type2tid.append(1)
    type2edge.append([2, 0])

    type2tid = torch.LongTensor(type2tid).to(device)
    type2edge = torch.LongTensor(type2edge).to(device)
    return type2tid, type2edge

def normalize_grad_by_barycoords(grad_over_triangles, barycoords):
    barycoords = barycoords.detach()
    grad_over_triangles = grad_over_triangles / barycoords.unsqueeze(-1)
    return grad_over_triangles


def normalize_grad_by_edge_coefs(grad_over_edges, coeffs):
    coeffs = coeffs.detach()
    coeffs = torch.stack([1 - coeffs, coeffs], dim=1)
    grad_over_edges = grad_over_edges / coeffs.unsqueeze(-1)


    return grad_over_edges

def separate_face_ids(face_id_pairs, collision_types):
    collision_types = collision_types[:, 0]

    type2tid, type2edge = make_type2ind_triedge(collision_types.device)

    type2tid_id = type2tid[collision_types].unsqueeze(-1) 
    face_ids_gathered = torch.gather(face_id_pairs, 1, type2tid_id)
    face_ids_remaining = torch.gather(face_id_pairs, 1, 1 - type2tid_id)

    return face_ids_gathered[..., 0], face_ids_remaining[..., 0]

def binmasks2onehot(binmasks):

    type_bmask = binmasks[:, 0]
    loop_bmask = binmasks[:, 1]

    type_ohot = []
    loop_ohot = []

    for i in range(6):
        tohot = type_bmask // 2**i % 2
        type_ohot.append(tohot.bool())

        lohot = loop_bmask // 2**i % 2
        loop_ohot.append(lohot.bool())

    type_ohot = torch.stack(type_ohot, dim=-1)
    loop_ohot = torch.stack(loop_ohot, dim=-1)

    return type_ohot, loop_ohot

def get_triedge_candidates(verts, faces):
    triangles = verts[faces].unsqueeze(0).contiguous()

    bboxes, tree = torch_collisions.bvh(triangles)
    collision_tensor, binmasks = torch_collisions.find_triangle_edge_candidates2(bboxes, tree, triangles, max_collisions=64)
    collision_tensor = collision_tensor[0]
    binmasks = binmasks[0]


    mask = collision_tensor[:, 0] != -1
    collision_tensor = collision_tensor[mask]
    binmasks = binmasks[mask]

    typemasks, loopmasks = binmasks2onehot(binmasks)



    return triangles[0], collision_tensor, typemasks, loopmasks

def compute_edge_coeffs(colliding_edges, triangle_normal, triangle_point, detach_aux=True):

    triangle_normal = triangle_normal.detach()

    if detach_aux:
        triangle_point = triangle_point.detach()

    e_edge = colliding_edges[:, 1] - colliding_edges[:, 0]
    en_dot = (e_edge * triangle_normal).sum(-1)

    d = (triangle_normal * triangle_point).sum(-1)
    t = (d - (triangle_normal * colliding_edges[:, 0]).sum(-1)) / en_dot

    t[en_dot.abs() < 1e-15] = -1

    return t

def compute_barycoords(colliding_triangles, intersection_points, detach_aux=True):
    if detach_aux:
        intersection_points = intersection_points.detach()

    t_edge0 = colliding_triangles[:, 1] - colliding_triangles[:, 0]
    t_edge2 = colliding_triangles[:, 2] - colliding_triangles[:, 0]


    ep = intersection_points - colliding_triangles[:, 0]
    d00 = (t_edge0 * t_edge0).sum(-1)
    d01 = (t_edge0 * t_edge2).sum(-1)
    d11 = (t_edge2 * t_edge2).sum(-1)
    denom = d00 * d11 - d01 * d01

    d20 = (ep * t_edge0).sum(-1)
    d21 = (ep * t_edge2).sum(-1)

    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1. - v - w

    barycoords = torch.stack([u, v, w], dim=-1)

    barycoords[denom.abs() < 1e-15] = -1

    return barycoords

def compute_normals(triangles):
    t_edge1 = triangles[:, 0] - triangles[:, 1]
    t_edge2 = triangles[:, 2] - triangles[:, 1]

    triangle_normal = torch.linalg.cross(t_edge2, t_edge1)
    triangle_normal = torch.nn.functional.normalize(triangle_normal, dim=-1)

    return triangle_normal

def compute_coeffs(colliding_edges, colliding_triangles, apply_mask=True, detach_aux_bary=True, detach_aux_edges=True):
    triangle_normal = compute_normals(colliding_triangles)

    edge_coeffs = compute_edge_coeffs(colliding_edges, triangle_normal, colliding_triangles.mean(1), detach_aux=detach_aux_edges)

    intersection_points = colliding_edges[:, 0] + edge_coeffs[:, None] * (colliding_edges[:, 1] - colliding_edges[:, 0])
    barycoords = compute_barycoords(colliding_triangles, intersection_points, detach_aux=detach_aux_bary)

    barycoords_mask = (barycoords >= 0).all(-1)
    edgecoef_mask = (edge_coeffs >= 0) & (edge_coeffs <= 1)
    valid_mask = barycoords_mask & edgecoef_mask

    if apply_mask:
        edge_coeffs = edge_coeffs[valid_mask]
        barycoords = barycoords[valid_mask]

    return edge_coeffs, barycoords, valid_mask

def get_triedge_collisions(verts, faces):
    triangles, face_id_pairs, typemasks, loopmasks = get_triedge_candidates(verts, faces)

    types_row = torch.arange(6).to(typemasks.device)[None]

    type_ids = typemasks * types_row
    type_ids[~typemasks] = -1

    type_ids_flat = type_ids.reshape(-1)
    loopmask_flat = loopmasks.reshape(-1)
    facepair_ids_flat = face_id_pairs.unsqueeze(1).repeat(1, 6, 1).reshape(-1, 2)
    valid_candidates_mask = type_ids_flat != -1

    collision_quads = torch.cat([facepair_ids_flat, type_ids_flat[..., None], loopmask_flat[..., None]], dim=-1)
    collision_quads = collision_quads[valid_candidates_mask]


    loop_mask = collision_quads[:, -1] == 1

    collision_quads_loop = collision_quads[loop_mask]
    collision_quads_noloop = collision_quads[~loop_mask]


    colliding_edges_all, triangles_colliding_edges_all, colliding_triangles_all = get_colliding_primitives(
        collision_quads_noloop, triangles)

    edge_coeffs, barycoords, valid_pen_mask = compute_coeffs(colliding_edges_all, colliding_triangles_all)
    collision_quads_noloop = collision_quads_noloop[valid_pen_mask]


    anv_mask1 = torch.zeros_like(valid_candidates_mask).bool()
    anv_mask2 = torch.zeros_like(loop_mask).bool()
    anv_mask3 = torch.zeros_like(valid_pen_mask).bool()

    anv_mask3[valid_pen_mask] = True
    anv_mask2[~loop_mask] = anv_mask3
    anv_mask1[valid_candidates_mask] = anv_mask2

    type_ids_flat_valnl = torch.ones_like(type_ids_flat)*-1
    type_ids_flat_valnl[anv_mask1] = collision_quads_noloop[:, 2]
    type_ids_flat_valnl = type_ids_flat_valnl.reshape(-1, 6)
    rows_valid = (type_ids_flat_valnl >= 0).sum(dim=-1) > 0


    al_mask1 = torch.zeros_like(valid_candidates_mask).bool()
    al_mask2 = torch.zeros_like(loop_mask).bool()
    al_mask2[loop_mask] = True
    al_mask1[valid_candidates_mask] = al_mask2

    type_ids_flat_all = torch.ones_like(type_ids_flat)*-1
    type_ids_flat_all[al_mask1] = collision_quads_loop[:, 2]
    type_ids_flat_all = type_ids_flat_all.reshape(-1, 6)
    type_ids_flat_all[~rows_valid] = -1

    type_ids_all = type_ids_flat_valnl
    type_ids_all[type_ids_flat_all >= 0] = type_ids_flat_all[type_ids_flat_all >= 0]
    loopmasks[type_ids_all < 0] = False

    type_ids_all = type_ids_all[rows_valid]
    loopmasks = loopmasks[rows_valid]
    face_id_pairs = face_id_pairs[rows_valid]

    # remove triangle-triangle collisions with more or fewer than 2 intersections
    npen_per_row = (type_ids_all >= 0).sum(-1)
    rows_valid2 = npen_per_row == 2

    type_ids_all = type_ids_all[rows_valid2]
    loopmasks = loopmasks[rows_valid2]
    face_id_pairs = face_id_pairs[rows_valid2]

    type_ids_all_mask = type_ids_all >= 0
    coll_type_ids = torch.where(type_ids_all_mask)[1].view(-1, 2)

    type_ids_all = torch.gather(type_ids_all, 1, coll_type_ids)
    loopmasks = torch.gather(loopmasks, 1, coll_type_ids)

    return triangles, face_id_pairs, type_ids_all, loopmasks


def get_colliding_primitives_selected(face_id_pairs, type_ids, loopmask, triangles, faces):

    type2tid, type2edge = make_type2ind_triedge(triangles.device)

    type2tid_id = type2tid[type_ids].unsqueeze(-1)  # .repeat(1, 3)
    type2edge = type2edge[type_ids]
    face_ids_colliding_edges = torch.gather(face_id_pairs, 1, type2tid_id)
    face_ids_colliding_triangles = torch.gather(face_id_pairs, 1, 1 - type2tid_id)

    triangles_colliding_edges = triangles[face_ids_colliding_edges][:, 0]
    colliding_triangles = triangles[face_ids_colliding_triangles][:, 0]

    colliding_edges = torch.gather(triangles_colliding_edges, 1, type2edge.unsqueeze(-1).repeat(1, 1, 3))


    triangles_colliding_edges_ids = faces[face_ids_colliding_edges][:, 0]
    colliding_edges_ids = torch.gather(triangles_colliding_edges_ids, 1, type2edge)
    loop_edges_ids = colliding_edges_ids[loopmask]
    loop_node_ids =  loop_edges_ids[:, 0]

    return colliding_edges, triangles_colliding_edges, colliding_triangles, loop_node_ids



def compute_gradient_edge_coeffs(colliding_edges, edge_coeffs, loopmask, intersecting_points_loop, normalize=False):
    intersection_points = colliding_edges[:, 0] + edge_coeffs[:, None] * (colliding_edges[:, 1] - colliding_edges[:, 0])
    intersection_points[loopmask] = intersecting_points_loop

    intersection_segments = intersection_points.view(-1, 2, 3)
    segment_lengths_sq = (intersection_segments[:, 0] - intersection_segments[:, 1]).pow(2).sum(-1)
    loss = segment_lengths_sq.sum()
    grad_edge_coeffs, = torch.autograd.grad(loss, edge_coeffs)
    segment_lengths_all = segment_lengths_sq.detach().sqrt().unsqueeze(-1).repeat(1, 2).view(-1)

    if normalize:
        grad_edge_coeffs = grad_edge_coeffs / segment_lengths_all

    return grad_edge_coeffs, segment_lengths_all

def compute_gradient_barycoords(colliding_triangles, barycoords, loopmask, intersecting_points_loop, normalize=False):
    intersection_points = (colliding_triangles * barycoords.unsqueeze(-1)).sum(1)
    intersection_points[loopmask] = intersecting_points_loop

    intersection_segments = intersection_points.view(-1, 2, 3)
    segment_lengths_sq = (intersection_segments[:, 0] - intersection_segments[:, 1]).pow(2).sum(-1)
    loss = segment_lengths_sq.sum()
    grad_barycoords, = torch.autograd.grad(loss, barycoords)
    segment_lengths_all = segment_lengths_sq.detach().sqrt().unsqueeze(-1).repeat(1, 2).view(-1)

    if normalize:
        grad_barycoords = grad_barycoords / segment_lengths_all[:, None] #/ edge_lengths
    return grad_barycoords, segment_lengths_all

def compute_icloss_edges(colliding_edges, edge_coeffs, loopmask, intersecting_points_loop, detach_coords=False, detach_coeffs=False):

    if detach_coords:
        colliding_edges = colliding_edges.detach()

    if detach_coeffs:
        edge_coeffs = edge_coeffs.detach()


    intersection_points = colliding_edges[:, 0] + edge_coeffs[:, None] * (colliding_edges[:, 1] - colliding_edges[:, 0])
    intersection_points[loopmask] = intersecting_points_loop

    intersection_segments = intersection_points.view(-1, 2, 3)
    segment_lengths_sq = (intersection_segments[:, 0] - intersection_segments[:, 1]).pow(2).sum(-1)



    segment_lengths_sq[segment_lengths_sq != segment_lengths_sq] = 0
    loss = segment_lengths_sq.sum()


    return loss

def compute_icloss_faces(colliding_triangles, barycoords, loopmask, intersecting_points_loop, detach_coords=False, detach_coeffs=False):
    if detach_coords:
        colliding_triangles = colliding_triangles.detach()

    if detach_coeffs:
        barycoords = barycoords.detach()

    intersection_points = (colliding_triangles * barycoords.unsqueeze(-1)).sum(1)
    intersection_points[loopmask] = intersecting_points_loop

    intersection_segments = intersection_points.view(-1, 2, 3)

    segment_lengths_sq = (intersection_segments[:, 0] - intersection_segments[:, 1]).pow(2).sum(-1)


    segment_lengths_sq[segment_lengths_sq != segment_lengths_sq] = 0

    loss = segment_lengths_sq.sum()

    return loss

def get_intersection_edges(faces, face_id_pairs, collision_types, coeffs=None, loop_mask=None):

    intersection_edges = get_tri_edge_intersection_edges(faces, face_id_pairs, collision_types)
    edges_argsort = torch.argsort(intersection_edges, dim=-1)
    to_flip = edges_argsort[:, 0] == 1
    to_flip[loop_mask]
    intersection_edges = torch.gather(intersection_edges, 1, edges_argsort)

    if coeffs is not None:
        coeffs = coeffs.clone()
        coeffs[to_flip] = 1. - coeffs[to_flip]
        return intersection_edges, coeffs
    else:
        return intersection_edges
    
def create_additional_node_ids(intersection_edges, face_ids_remaining, loopmask):
    intersection_triples = torch.cat([intersection_edges, face_ids_remaining[:, None]], -1)
    triplets_unique, inverse_ids, counts = torch.unique(intersection_triples, return_counts=True, return_inverse=True,
                                                        dim=0)
    
    inverse_ids[loopmask] = -1
    return inverse_ids

def split_contours(type_ids, loopmask_flat, additional_node_ids):
    tti = torch.arange(additional_node_ids.shape[0]//2).to(additional_node_ids.device)
    tti = tti[None].repeat(1, 2).reshape(-1)

    colli =  torch.arange(additional_node_ids.shape[0]).to(additional_node_ids.device)

    colli_stack = []
    colli_processed = set()
    contour_ids = torch.ones_like(additional_node_ids).long() * -2


    loop_coli = torch.where(loopmask_flat)[0]
    for lcoli in loop_coli:
        colli_processed.add(lcoli.item())
    contour_ids[loop_coli] = -1


    ani2coli = defaultdict(list)
    for i, ani in enumerate(additional_node_ids):
        ani = ani.item()
        ani2coli[ani].append(i)



    last_label = 0
    while len(colli_stack) > 0 or len(colli_processed) < colli.shape[0]:

        if len(colli_stack) == 0:
            nonproc = torch.where(contour_ids == -2)[0]
            colli_cur = nonproc[0].item()
            contour_ids[colli_cur] = last_label
            last_label += 2
        else:
            colli_cur = colli_stack.pop()

        if colli_cur in colli_processed:
            continue
        colli_processed.add(colli_cur)

        curr_contour = contour_ids[colli_cur]
        opposite_contour = curr_contour + 1 if curr_contour % 2 == 0 else curr_contour - 1

        curr_ani = additional_node_ids[colli_cur].item()

        if curr_ani == -1:
            continue

        coli_ani = ani2coli[curr_ani]
        for coli in coli_ani:
            if coli != colli_cur:
                contour_ids[coli] = curr_contour

                if coli not in colli_processed:
                    assert type(coli) == int
                    colli_stack.append(coli)

        curr_tti = colli_cur // 2
        same_tti_coli = colli_cur + 1 if colli_cur % 2 == 0 else colli_cur - 1
        same_tti_ani = additional_node_ids[same_tti_coli].item()
        assert type(same_tti_coli) == int

        if same_tti_ani == -1:
            continue

        curr_type_ids = type_ids[curr_tti] // 3
        if curr_type_ids[0] == curr_type_ids[1]:
            contour_ids[same_tti_coli] = curr_contour

            if same_tti_coli not in colli_processed:
                colli_stack.append(same_tti_coli)
        else:
            contour_ids[same_tti_coli] = opposite_contour
            if same_tti_coli not in colli_processed:
                colli_stack.append(same_tti_coli)


    return contour_ids


def create_nonmultipass_edges(intersection_edges, additional_node_ids):
    if intersection_edges.shape[0] == 0:
        return torch.zeros(0, 2).long().to(intersection_edges.device)
    e1 = torch.stack([intersection_edges[:, 0], additional_node_ids], dim=1)
    e2 = torch.stack([intersection_edges[:, 1], additional_node_ids], dim=1)
    new_edges = torch.cat([e1, e2], dim=0)

    return new_edges


def create_multipass_edges(intersection_edges, edge_coefs, face_ids_gathered, additional_node_ids):
    if intersection_edges.shape[0] == 0:
        return torch.zeros(0, 2).long().to(intersection_edges.device)

    intersection_triples = torch.cat([intersection_edges, face_ids_gathered[:, None]], -1)
    _, inverse_ids, _ = torch.unique(intersection_triples, return_counts=True, return_inverse=True,
                                                        dim=0)


    new_edges = []
    n_edges = inverse_ids.max() + 1
    for i in range(n_edges):
        edge_mask = inverse_ids == i

        ie = intersection_edges[edge_mask]
        coefs = edge_coefs[edge_mask]
        ani = additional_node_ids[edge_mask]

        v_from = ie[0,0].item()
        v_to = ie[0,1].item()

        coefs_argsort = torch.argsort(coefs)
        ani_sorted = ani[coefs_argsort]


        new_edges.append((v_from, ani_sorted[0]))
        new_edges.append((v_to, ani_sorted[1]))

        for i in range(len(ani_sorted)-1):
            if ani_sorted[i] != ani_sorted[i+1]:
                new_edges.append((ani_sorted[i], ani_sorted[i+1]))

    new_edges = torch.LongTensor(new_edges).to(intersection_edges.device)
    return new_edges

def build_graph_with_intersections(faces, intersection_edges, loopmask, face_ids_gathered, face_ids_remaining, edge_coeffs, additional_node_ids):
    
    edge_coeffs = edge_coeffs.detach()
    all_edges = triangles_to_edges(faces.unsqueeze(0), two_way=False).squeeze(0).T

    ani_offset = all_edges.max() + 1
    additional_node_ids = additional_node_ids + ani_offset

    id_offset = additional_node_ids.max() + 1

    all_edges_hashed = all_edges[:, 0] * id_offset + all_edges[:, 1]
    intersection_edges_hashed = intersection_edges[:, 0] * id_offset + intersection_edges[:, 1]
    intersection_edges_hashed_noloop = torch.unique(intersection_edges_hashed[~loopmask])

    edges_no_isect = all_edges[~torch.isin(all_edges_hashed, intersection_edges_hashed_noloop)]


    intersection_edges_noloop = intersection_edges[~loopmask]
    face_ids_gathered_noloop = face_ids_gathered[~loopmask]
    face_ids_remaining_noloop = face_ids_remaining[~loopmask]
    edge_coeffs_noloop = edge_coeffs[~loopmask]
    additional_node_ids_noloop = additional_node_ids[~loopmask]


    multipass_mask_all = make_multipass_mask(intersection_edges_noloop, face_ids_gathered_noloop)
    nmp_mask = ~multipass_mask_all

    nonmultipass_edges = create_nonmultipass_edges(intersection_edges_noloop[nmp_mask], additional_node_ids_noloop[nmp_mask])
    nonmultipass_edges  = torch.unique(nonmultipass_edges, dim=0)

    multipass_edges = create_multipass_edges(intersection_edges_noloop[multipass_mask_all], edge_coeffs_noloop[multipass_mask_all], 
                                             face_ids_gathered_noloop[multipass_mask_all], additional_node_ids_noloop[multipass_mask_all])
    multipass_edges = torch.unique(multipass_edges, dim=0)

    full_graph_edges = torch.cat([edges_no_isect, nonmultipass_edges, multipass_edges], dim=0)

    return full_graph_edges, ani_offset



    
def replace_ani_with_surface_labels(full_graph_edges, surface_labels, additional_node_ids, ani_offset):

    full_graph_edges = full_graph_edges.clone()
    sli_offset = full_graph_edges.max() + 1
    surface_labels = surface_labels + sli_offset


    graph_ani_mask = full_graph_edges >= ani_offset
    graph_ani = full_graph_edges[graph_ani_mask]
    N_coll = surface_labels.shape[0]

    additional_node_ids = additional_node_ids + ani_offset
    ani_wg = torch.cat([additional_node_ids, graph_ani])
    ani_wg_unique, ani_wg_inverse = torch.unique(ani_wg, return_inverse=True)

    ani_inverse = ani_wg_inverse[:N_coll]
    ani2sli = torch_scatter.scatter(surface_labels, ani_inverse, reduce='max')

    ani_g_inverse = ani_wg_inverse[N_coll:]

    sli_g = ani2sli[ani_g_inverse]
    full_graph_edges[graph_ani_mask] = sli_g

    asort = torch.argsort(full_graph_edges, dim=-1)
    full_graph_edges = torch.gather(full_graph_edges, 1, asort)

    full_graph_edges = torch.unique(full_graph_edges, dim=0)

    return full_graph_edges, sli_offset


def make_edge_df(edges, with_inverse=False):
    if with_inverse:
        edges = torch.cat([edges, edges[:, [1,0]]], dim=0)

    df = cudf.DataFrame()
    df['src'] = edges[:, 0].cpu().numpy()
    df['dst'] = edges[:, 1].cpu().numpy()
    return df


def replace_node_ids_with_components(full_graph_edges, loop_node_ids, sli_offset):
    full_graph_edges = full_graph_edges.clone()
    components_offset = full_graph_edges.max().item() + 1

    edges_noloop = full_graph_edges[~torch.isin(full_graph_edges, loop_node_ids).any(dim=-1)]
    edges_nosl = edges_noloop[(edges_noloop < sli_offset).all(dim=-1)]

    edges_df = make_edge_df(edges_nosl)
    graph_contour = cugraph.Graph()
    graph_contour.from_cudf_edgelist(edges_df, source='src', destination='dst')

    components = cugraph.connected_components(graph_contour)
    components = torch.LongTensor(components.to_numpy()).to(full_graph_edges.device)


    clabels_unique, clables, counts = torch.unique(components[:, 1], return_inverse=True, return_counts=True)
    node_ids = components[:, 0]

    N_nodes = node_ids.shape[0]

    ni2counts = torch_scatter.scatter(counts[clables], clables, reduce='max')
    ni2count_dict = {i+components_offset: c.item() for i, c in enumerate(ni2counts)}
    ni2nodes_dict = {i+components_offset: node_ids[clables==i] for i in range(clables.max().item()+1)}


    graph_replace_mask = torch.isin(full_graph_edges, node_ids)
    graph_replace = full_graph_edges[graph_replace_mask]

    ni_wg = torch.cat([node_ids, graph_replace])
    ni_wg_unique, ni_wg_inverse = torch.unique(ni_wg, return_inverse=True)

    ni_inverse = ni_wg_inverse[:N_nodes]
    ni2ci = torch_scatter.scatter(clables, ni_inverse, reduce='max')

    ni_g_inverse = ni_wg_inverse[N_nodes:]

    ci_g = ni2ci[ni_g_inverse]
    ci_g = ci_g + components_offset

    full_graph_edges[graph_replace_mask] = ci_g

    asort = torch.argsort(full_graph_edges, dim=-1)
    full_graph_edges = torch.gather(full_graph_edges, 1, asort)

    full_graph_edges = torch.unique(full_graph_edges, dim=0)

    return full_graph_edges, ni2count_dict, ni2nodes_dict, components_offset


def renumerate_node_ids(full_graph_edges, sli_offset, components_offset, loop_offset, component_counts):
    full_graph_edges = full_graph_edges.clone()
    node_mask = (full_graph_edges < sli_offset) | (full_graph_edges >= components_offset)

    node_ids = full_graph_edges[node_mask]

    nodes_old_unique, node_ids_new = torch.unique(node_ids, return_inverse=True)

    node_id_new2old = torch_scatter.scatter(node_ids, node_ids_new, reduce='max')
    full_graph_edges[node_mask] = node_ids_new 
    new_sl_offset = node_ids_new.max().item() + 1

    component_counts_new = torch.zeros_like(node_id_new2old)
    component_is_loop = torch.zeros_like(node_id_new2old).bool()
    for nid_new, nid_old in enumerate(node_id_new2old):
        nid_old = nid_old.item()

        is_loop = False

        if nid_old in component_counts:
            component_counts_new[nid_new] = component_counts[nid_old]
        else:
            if nid_old >= loop_offset:
                is_loop = True
            component_counts_new[nid_new] = 1

        component_is_loop[nid_new] = is_loop

    sli_mask = (full_graph_edges >= sli_offset) & (full_graph_edges < components_offset)
    sli_old = full_graph_edges[sli_mask]

    sli_old_unique, sli_new = torch.unique(sli_old, return_inverse=True)
    sli_new += new_sl_offset



    full_graph_edges[sli_mask] = sli_new

    self_edges = full_graph_edges[:, 0] == full_graph_edges[:, 1]
    full_graph_edges = full_graph_edges[~self_edges]

    return full_graph_edges, component_counts_new, component_is_loop, new_sl_offset, nodes_old_unique, sli_old_unique

def renumerate_loop_nodes(full_graph_edges, loop_node_ids):
    loop_offset = full_graph_edges.max().item() + 1

    if loop_node_ids.shape[0] == 0:
        return full_graph_edges, loop_offset
    


    loop_edges = full_graph_edges[torch.isin(full_graph_edges, loop_node_ids).any(dim=-1)]

    graph = nx.Graph()
    graph.add_edges_from(loop_edges.cpu().numpy())
    graph = nx.subgraph(graph, loop_node_ids.cpu().numpy())

    ccomponents = list(nx.connected_components(graph))
    nid2lcomp = []

    for i, cc in enumerate(ccomponents):
        for nid in cc:
            nid2lcomp.append((nid, i))

    nid2lcomp = torch.LongTensor(nid2lcomp).to(full_graph_edges.device)

    nid, lcomp = nid2lcomp[:, 0], nid2lcomp[:, 1]

    graph_replace_mask = torch.isin(full_graph_edges, loop_node_ids)
    graph_replace = full_graph_edges[graph_replace_mask]

    nid_wg = torch.cat([nid, graph_replace])
    nid_wg_unique, nid_wg_inverse = torch.unique(nid_wg, return_inverse=True)
    N_loop = nid.shape[0]

    nid_inverse = nid_wg_inverse[:N_loop]
    nid2lcomp = torch_scatter.scatter(lcomp, nid_inverse, reduce='max')

    nid_g_inverse = nid_wg_inverse[N_loop:]

    lcomp_g = nid2lcomp[nid_g_inverse]

    lcomp_g = lcomp_g + loop_offset
    full_graph_edges[graph_replace_mask] = lcomp_g

    return full_graph_edges, loop_offset

def compute_conponent_size(component, component_counts, sli_offset):
    c = 0
    for node in component:
        if node < sli_offset:
            c += component_counts[node]

    return c

def check_encompassed(graph, sl, component_counts, sli_offset):
    graph = graph.copy()
    graph.remove_node(sl)

    ccomponents = list(nx.connected_components(graph))
    if len(ccomponents) != 2:
        return set(), set()

    csizes = [compute_conponent_size(cc, component_counts, sli_offset) for cc in ccomponents]
    if csizes[0] == csizes[1]:
        return set(), set()
    
    smaller_component = ccomponents[0] if csizes[0] < csizes[1] else ccomponents[1]

    enc_nodes = []
    enc_contours = []

    for node in smaller_component:
        if node >= sli_offset:
            enc_contours.append(node)
        else:
            enc_nodes.append(node)


    return set(enc_nodes), set(enc_contours)

def cleanup_encompassed(sl2encnodes, sl2enccontours):
    encompassed_edges = []
    for sl, enc_contours in sl2enccontours.items():
        for contour in enc_contours:
            encompassed_edges.append((contour, sl))


    graph = nx.DiGraph()
    graph.add_edges_from(encompassed_edges)

    while True:
        leaves = [n for n in graph.nodes() if graph.in_degree(n) == 0 and graph.out_degree(n) > 0]
        if len(leaves) == 0:
            break
        for leaf in leaves:
            parents = list(graph.successors(leaf))
            graph.remove_node(leaf)


            for parent in parents:
                sl2encnodes[parent].update(sl2encnodes[leaf])
                sl2enccontours[parent].update(sl2enccontours[leaf])

            
            sl2encnodes.pop(leaf)
            sl2enccontours.pop(leaf)

    return sl2encnodes, sl2enccontours


def mark_encompassed_clusters(full_graph_components, component_counts, component_is_loop, sli_offset, sli_old, sli_offset_old):
    sl_ids = torch.unique(full_graph_components[full_graph_components >= sli_offset])
    loop_ids = torch.where(component_is_loop)[0]

    graph = nx.Graph()
    graph.add_edges_from(full_graph_components.cpu().numpy())
    graph.remove_nodes_from(loop_ids.cpu().numpy())

    sl2encnodes = dict()
    sl2enccontours = dict()

    ccomponents = list(nx.connected_components(graph))
    for ccomponent in ccomponents:
        component_graph = nx.Graph(graph.subgraph(ccomponent))

        for node in ccomponent:
            if node in sl_ids:
                enc_nodes, enc_contours = check_encompassed(component_graph, node, component_counts, sli_offset)
                sl2encnodes[node] = enc_nodes
                sl2enccontours[node] = enc_contours

    sl2encnodes_new, sl2enccontours = cleanup_encompassed(sl2encnodes, sl2enccontours)
    sl2encnodes = sl2encnodes_new

    for sl in list(sl2encnodes.keys()):
        if len(sl2encnodes[sl]) == 0 and len(sl2enccontours[sl]) == 0:    
            sl2encnodes.pop(sl)
            sl2enccontours.pop(sl)
            continue


    return sl2encnodes, sl2enccontours


def reset_renumerations(sl2encnodes, sl2enccontours, sli_old, nodes_old, component_nodes, sli_offset_new, sli_offset_old, n_verts):
    sl2enccontours_new = dict()

    verts_encompassed_labels = torch.ones(n_verts).long().to(sli_old.device) * -1


    k_old_deb = []
    k_new_deb = []

    for k in sl2encnodes.keys():
        k_new = sli_old[k-sli_offset_new] - sli_offset_old
        k_new = k_new.item()

        k_old_deb.append(k)
        k_new_deb.append(k_new)

        encnodes = torch.LongTensor(list(sl2encnodes[k])).to(sli_old.device)
        enccontours = torch.LongTensor(list(sl2enccontours[k])).to(sli_old.device)

        enccontours_new = sli_old[enccontours-sli_offset_new] - sli_offset_old
        encnodes_new = nodes_old[encnodes] #- sli_offset_old
        sl2enccontours_new[k_new] = enccontours_new

        nodes_encompassed = []

        for encnode in encnodes_new:
            if encnode.item() in component_nodes:
                nodes_encompassed.append(component_nodes[encnode.item()])
            else:
                nodes_encompassed.append(encnode[None,])

        if len(nodes_encompassed) == 0:
            continue
        nodes_encompassed = torch.cat(nodes_encompassed, dim=0)
        verts_encompassed_labels[nodes_encompassed] = k_new
    return verts_encompassed_labels, sl2enccontours_new


def mark_encompassed(full_graph_edges, surface_labels, additional_node_ids, loopmask, loop_node_ids, n_verts, ani_offset):
    graph_slabels, sli_offset = replace_ani_with_surface_labels(full_graph_edges, surface_labels[~loopmask], additional_node_ids[~loopmask],  ani_offset)
    graph_components, component_counts, component_nodes, components_offset = replace_node_ids_with_components(graph_slabels, loop_node_ids, sli_offset)
    graph_components, loop_offset = renumerate_loop_nodes(graph_components, loop_node_ids)
    graph_components, component_counts, component_is_loop, sli_offset_new, nodes_old, sli_old = renumerate_node_ids(graph_components, sli_offset, components_offset,
                                                                                            loop_offset, component_counts)
    sl2encnodes, sl2enccontours = mark_encompassed_clusters(graph_components, component_counts, component_is_loop, sli_offset_new, sli_old, sli_offset)
    nodes_encompassed_labels, sl2enccontours_new = reset_renumerations(sl2encnodes, sl2enccontours, sli_old, nodes_old, component_nodes, sli_offset_new, sli_offset, n_verts)
    

    encompassed_contours = list(sl2enccontours_new.values())
    if len(encompassed_contours) == 0:
        encompassed_contours = torch.zeros(0).long().to(surface_labels.device)
    else:
        encompassed_contours = torch.cat(encompassed_contours, dim=0)

    closed_contours = [k for k in sl2encnodes.keys() if len(sl2encnodes[k]) > 0 or len(sl2enccontours[k]) > 0]
    closed_contours = [sli_old[k-sli_offset_new] - sli_offset for k in closed_contours]
    closed_contours = torch.LongTensor(closed_contours).to(surface_labels.device)

    return nodes_encompassed_labels, encompassed_contours, closed_contours

def average_grads_across_surfaces(grad_ic_ce, grad_ic_ct, surface_labels, nodes_encompassed_labels, segment_lengths, cl_weighting=False, reduce='sum'):
    n_surfaces = surface_labels.max().item() + 1

    if n_surfaces % 2 == 1:
        n_surfaces += 1


    keep_mask = surface_labels != -1

    grad_edges_avg = grad_ic_ce.mean(1)[keep_mask]
    grad_faces_avg = grad_ic_ct.mean(1)[keep_mask]
    segment_lengths = segment_lengths[keep_mask]

    surface_labels = surface_labels[keep_mask]
    surface_labels_mod = surface_labels % 2
    surface_labels_flipped = surface_labels - surface_labels_mod + ( 1- surface_labels_mod)

    grad_edges_gathered = torch.zeros(n_surfaces, 3).to(grad_ic_ce.device)
    grad_faces_gathered = torch.zeros(n_surfaces, 3).to(grad_ic_ct.device)


    torch_scatter.scatter(grad_edges_avg, surface_labels, out=grad_edges_gathered, dim=0, reduce=reduce)
    torch_scatter.scatter(grad_faces_avg, surface_labels, out=grad_faces_gathered, dim=0, reduce=reduce)


    grad_faces_gathered = grad_faces_gathered.reshape(-1, 2, 3)
    grad_faces_gathered  = grad_faces_gathered[:, [1,0]].reshape(-1, 3)

    grad_sum_gathered = (grad_edges_gathered + grad_faces_gathered) / 2

    if cl_weighting:
        segment_lengths = segment_lengths[:, None]
        contour_lengths = torch.zeros(n_surfaces, 1).to(grad_ic_ct.device)
        torch_scatter.scatter(segment_lengths, surface_labels, out=contour_lengths, dim=0)


        grad_sum_gathered = grad_sum_gathered * contour_lengths


    # grad_sum_gathered = grad_edges_gathered
    grad_sum_disrtibuted = grad_sum_gathered[surface_labels]

    grad_edges = grad_sum_gathered[surface_labels]
    grad_faces = grad_sum_gathered[surface_labels_flipped]

    n_nodes = nodes_encompassed_labels.shape[0]
    nodes_encompassed_grad = torch.zeros((n_nodes, 3)).float().to(grad_ic_ce.device)

    encompassed_mask = nodes_encompassed_labels != -1
    labels_to_insert = nodes_encompassed_labels[encompassed_mask]

    nodes_encompassed_grad[encompassed_mask] = grad_sum_gathered[labels_to_insert]

    n_collisions = grad_ic_ce.shape[0]
    grad_edges_all = torch.zeros((n_collisions, 3)).float().to(grad_ic_ce.device)
    grad_edges_all[keep_mask] = grad_edges

    grad_faces_all = torch.zeros((n_collisions, 3)).float().to(grad_ic_ce.device)
    grad_faces_all[keep_mask] = grad_faces


    return grad_edges_all, grad_faces_all, nodes_encompassed_grad


def make_node2contour(faces, face_ids_gathered, type_ids, surface_labels, nodes_encompassed_labels, n_verts):
    n_labels = surface_labels.max().item() + 1
    if n_labels % 2 == 1:
        n_labels += 1
    node2contour = torch.zeros(n_verts,n_labels).bool().to(faces.device)

    _, type2edge = make_type2ind_triedge(faces.device)


    for sl in torch.unique(surface_labels):
        if sl == -1:
            continue
        mask = surface_labels == sl
        face_ids = face_ids_gathered[mask]
        faces_selected = faces[face_ids]

        type2edge_selected = type2edge[type_ids[mask]]
        nodes_selected = torch.unique(torch.gather(faces_selected, 1, type2edge_selected))

        nodes_encompassed = nodes_encompassed_labels == sl

        node2contour[nodes_selected, sl] = True
        node2contour[nodes_encompassed, sl] = True

    return node2contour


def make_face2contour(face_ids_remaining, type_ids, surface_labels, nodes_encompassed_labels, faces):
    n_faces = faces.shape[0]
    n_labels = surface_labels.max().item() + 1
    if n_labels % 2 == 1:
        n_labels += 1

    face2contour = torch.zeros(n_faces, n_labels).bool().to(face_ids_remaining.device) 

    surface_labels_mod = surface_labels % 2
    # surface_labels_flipped = surface_labels - surface_labels_mod + ( 1- surface_labels_mod)

    surface_labels_flipped = surface_labels

    for sl in torch.unique(surface_labels):
        if sl == -1:
            continue
        mask = surface_labels == sl
        face_ids = face_ids_remaining[mask]
        faces_selected = face_ids

        sl_flipped = sl + 1 if sl % 2 == 0 else sl - 1
        nodes_encompassed = nodes_encompassed_labels == sl_flipped
        faces_encompassed = nodes_encompassed[faces].any(dim=-1)

        face2contour[faces_selected, sl] = True
        face2contour[faces_encompassed, sl] = True
    return face2contour

def compute_icontour_grad(verts, faces, cl_weighting=False):
    verts = verts.detach()
    verts.requires_grad = True

    triangles, face_id_pairs, type_ids, loopmasks = get_triedge_collisions(verts, faces)
    if face_id_pairs.shape[0] == 0:
        return torch.zeros_like(verts), dict()


    face_id_pairs_flat = face_id_pairs.unsqueeze(1).repeat(1, 2, 1).reshape(-1, 2)
    type_ids_flat = type_ids.reshape(-1)
    loopmasks_flat = loopmasks.reshape(-1)



    colliding_edges, triangles_colliding_edges, \
    colliding_triangles, loop_node_ids = get_colliding_primitives_selected(face_id_pairs_flat, type_ids_flat, loopmasks_flat, triangles, faces)
    intersecting_points_loop = verts[loop_node_ids].detach()


    all_edges = triangles_to_edges(faces.unsqueeze(0), two_way=False).squeeze(0).T
    is_loop_mask = torch.isin(all_edges, loop_node_ids).all(dim=-1)
    lledges = all_edges[is_loop_mask]
    lledges = torch.unique(lledges, dim=0)

    edge_coeffs, barycoords, _ = compute_coeffs(colliding_edges, colliding_triangles, apply_mask=False, detach_aux_edges=False)

    edge_coeffs = edge_coeffs * ~loopmasks_flat - 1 * loopmasks_flat
    barycoords = barycoords * ~loopmasks_flat[..., None] - 1 * loopmasks_flat[..., None]

    normalize = True
    grad_ic_ec, segment_lengths = compute_gradient_edge_coeffs(colliding_edges, edge_coeffs, loopmasks_flat,
                                              intersecting_points_loop, normalize=normalize)
    grad_ic_bary, _ = compute_gradient_barycoords(colliding_triangles, barycoords, loopmasks_flat,
                                               intersecting_points_loop, normalize=normalize)
    


    grad_ic_ce = torch.autograd.grad(edge_coeffs, colliding_edges, grad_ic_ec, retain_graph=True)[0]
    grad_ic_ct = torch.autograd.grad(barycoords, colliding_triangles, grad_ic_bary, retain_graph=True)[0]


    grad_ic_ce = normalize_grad_by_edge_coefs(grad_ic_ce, edge_coeffs)
    grad_ic_ct = normalize_grad_by_barycoords(grad_ic_ct, barycoords)


    intersection_edges, edge_coeffs_winv = get_intersection_edges(faces, face_id_pairs_flat, type_ids_flat[:, None], 
                                                                  coeffs=edge_coeffs, loop_mask=loopmasks_flat)
    face_ids_gathered, face_ids_remaining = separate_face_ids(face_id_pairs_flat, type_ids_flat[:, None])
    additional_node_ids = create_additional_node_ids(intersection_edges, face_ids_remaining, loopmasks_flat)

    surface_labels = split_contours(type_ids, loopmasks_flat, additional_node_ids)

    full_graph_edges, ani_offset = build_graph_with_intersections(faces, intersection_edges, loopmasks_flat, face_ids_gathered, face_ids_remaining, edge_coeffs_winv, additional_node_ids)
    nodes_encompassed_labels, encompassed_contours, closed_contours = mark_encompassed(full_graph_edges, surface_labels, additional_node_ids, loopmasks_flat, loop_node_ids, verts.shape[0], ani_offset)


    surface_labels[torch.isin(surface_labels, encompassed_contours)] = -1

    grad_edges_gathered, grad_faces_gathered, enclosed_nodes_grad = average_grads_across_surfaces(grad_ic_ce, grad_ic_ct, surface_labels,
                                                        nodes_encompassed_labels, segment_lengths, cl_weighting=cl_weighting)
    

    grad_edges_gathered = grad_edges_gathered.unsqueeze(1).repeat(1,3,1)
    grad_faces_gathered = grad_faces_gathered.unsqueeze(1).repeat(1,3,1)
    
    grad_ic_ec_verts = torch.autograd.grad(triangles_colliding_edges, verts, grad_edges_gathered, retain_graph=True)[0]
    grad_ic_bary_verts = torch.autograd.grad(colliding_triangles, verts, grad_faces_gathered, retain_graph=True)[0]

    total_grad_verts = grad_ic_ec_verts + grad_ic_bary_verts
    verts_zero_grad_mask = total_grad_verts.pow(2).sum(-1) == 0
    verts_enclosed_mask = nodes_encompassed_labels!=-1
    verts_add_grad_mask = verts_zero_grad_mask & verts_enclosed_mask
    total_grad_verts[verts_add_grad_mask] = enclosed_nodes_grad[verts_add_grad_mask]
    total_grad_verts *= -1

    node2contour = make_node2contour(faces, face_ids_gathered, type_ids_flat, surface_labels, nodes_encompassed_labels, verts.shape[0])
    face2contour = make_face2contour(face_ids_remaining, type_ids_flat, surface_labels, nodes_encompassed_labels, faces)

    closed_contours_mask = torch.zeros_like(node2contour[0]).bool()
    closed_contours_mask[closed_contours] = True 

    nodes_nenc_mask = node2contour[:, closed_contours]
    nodes_nenc_mask = ~nodes_nenc_mask.any(1) & node2contour.any(1)

    nodes_enclosed_or_nenc_mask = verts_enclosed_mask | nodes_nenc_mask
    nodes_enclosed_or_nenc_mask = nodes_enclosed_or_nenc_mask[:, None]

    total_grad_nnz_mask = total_grad_verts.pow(2).sum(-1) > 0
    node_in_contour_mask = node2contour.sum(1) > 0
    out_dict = dict()
    out_dict['node2contour'] = node2contour.detach()
    out_dict['face2contour'] = face2contour.detach()
    out_dict['enclosed_nodes_mask'] = verts_enclosed_mask.detach()
    out_dict['nodes_enclosed_or_nenc_mask'] = nodes_enclosed_or_nenc_mask.detach()


    out_dict['face_id_pairs'] = face_id_pairs.detach()
    out_dict['type_ids'] = type_ids.detach()
    out_dict['loopmasks'] = loopmasks.detach()


    return total_grad_verts.detach(), out_dict



def compute_igrad_loss(verts, faces, detach_coords=False, detach_coeffs=False, detach_aux_bary=True, detach_aux_edges=True, only_edgeloss=False):

    triangles, face_id_pairs, type_ids, loopmasks = get_triedge_collisions(verts, faces)
    if face_id_pairs.shape[0] == 0:
        return torch.FloatTensor([0]).to(verts.device)
    
    triangles = verts[faces]


    face_id_pairs_flat = face_id_pairs.unsqueeze(1).repeat(1, 2, 1).reshape(-1, 2)
    type_ids_flat = type_ids.reshape(-1)
    loopmasks_flat = loopmasks.reshape(-1)

    colliding_edges, triangles_colliding_edges, \
    colliding_triangles, loop_node_ids = get_colliding_primitives_selected(face_id_pairs_flat, type_ids_flat, loopmasks_flat, triangles, faces)
    intersecting_points_loop = verts[loop_node_ids].detach()

    edge_coeffs, barycoords, _ = compute_coeffs(colliding_edges, colliding_triangles, apply_mask=False, 
                                                detach_aux_bary=detach_aux_bary, detach_aux_edges=detach_aux_edges)

    edge_coeffs = edge_coeffs * ~loopmasks_flat - 1 * loopmasks_flat
    barycoords = barycoords * ~loopmasks_flat[..., None] - 1 * loopmasks_flat[..., None]

    edges_loss = compute_icloss_edges(colliding_edges, edge_coeffs, loopmasks_flat,
                                              intersecting_points_loop, detach_coords=detach_coords, detach_coeffs=detach_coeffs)


    faces_loss = compute_icloss_faces(colliding_triangles, barycoords, loopmasks_flat,
                                               intersecting_points_loop, detach_coords=detach_coords, detach_coeffs=detach_coeffs)
    

    if only_edgeloss:
        loss = edges_loss
    else:
        loss = (edges_loss + faces_loss) / 2

    return loss

