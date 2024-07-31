import torch
import warp as wp

from utils.warp_u.common import get_point_face_distance, vs_add, is_node_in_face, get_barycoords, get_point_plain_distance


wp.init()


@wp.kernel
def get_closest_faces(
        query_points: wp.array(dtype=wp.vec3),
        mesh_id: wp.uint64,
        num_query_points: int,
        result_face: wp.array(dtype=wp.int32),
        result_bary: wp.array(dtype=wp.vec2),
        max_distance: float
):
    tid = wp.tid()
    if tid >= num_query_points:
        return

    point = query_points[tid]

    face_index = int(-1)
    face_u = float(0.0)
    face_v = float(0.0)

    wp.mesh_query_point_no_sign(mesh_id, point, max_distance, face_index, face_u, face_v)

    result_face[tid] = face_index
    result_bary[tid] = wp.vec2(face_u, face_v)


@wp.func
def choose_closest_node(mesh_id: wp.uint64, face_id: wp.int32, point: wp.vec3):
    v0 = wp.mesh_get_point(mesh_id, face_id * 3 + 0)
    v1 = wp.mesh_get_point(mesh_id, face_id * 3 + 1)
    v2 = wp.mesh_get_point(mesh_id, face_id * 3 + 2)

    d0 = wp.length(v0 - point)
    d1 = wp.length(v1 - point)
    d2 = wp.length(v2 - point)

    if d0 < d1 and d0 < d2:
        return 0
    elif d1 < d2:
        return 1
    else:
        return 2
    
@wp.func
def choose_farthest_node(mesh_id: wp.uint64, face_id: wp.int32, point: wp.vec3):
    v0 = wp.mesh_get_point(mesh_id, face_id * 3 + 0)
    v1 = wp.mesh_get_point(mesh_id, face_id * 3 + 1)
    v2 = wp.mesh_get_point(mesh_id, face_id * 3 + 2)

    d0 = wp.length(v0 - point)
    d1 = wp.length(v1 - point)
    d2 = wp.length(v2 - point)

    if d0 > d1 and d0 > d2:
        return 0
    elif d1 > d2:
        return 1
    else:
        return 2

@wp.kernel
def get_closest_nodes_and_faces(
        query_points: wp.array(dtype=wp.vec3),
        mesh_id: wp.uint64,
        num_query_points: int,
        result_face: wp.array(dtype=wp.int32),
        result_node: wp.array(dtype=wp.int32),
        result_bary: wp.array(dtype=wp.vec2),
        max_distance: float
):


    tid = wp.tid()
    if tid >= num_query_points:
        return

    point = query_points[tid]

    face_index = int(-1)
    face_u = float(0.0)
    face_v = float(0.0)

    wp.mesh_query_point_no_sign(mesh_id, point, max_distance, face_index, face_u, face_v)


    result_face[tid] = face_index
    result_bary[tid] = wp.vec2(face_u, face_v)
    if face_index >= 0:

        
        node = choose_closest_node(mesh_id, face_index, point)
        result_node[tid] = wp.mesh_get_index(mesh_id, face_index * 3 + node)



# pytorch wrapper for get_closest_nodes_and_faces
def get_closest_nodes_and_faces_pt_dummmy(query_points, mesh_verts, mesh_faces, max_distance):
    faces_wp = wp.from_torch(mesh_faces.reshape(-1).int())
    mesh_verts_wp = wp.from_torch(mesh_verts.float().contiguous(), dtype=wp.vec3)
    query_points_wp = wp.from_torch(query_points.float().contiguous(), dtype=wp.vec3)

    mesh = wp.Mesh(
        points=mesh_verts_wp,
        indices=faces_wp,
    )


    num_query_points = query_points.shape[0]
    result_face_pt = torch.ones(num_query_points, dtype=torch.int32, device=query_points.device) * -1
    result_face_wp = wp.from_torch(result_face_pt.clone(), dtype=wp.int32)
    result_node_wp = wp.from_torch(result_face_pt.clone(), dtype=wp.int32)
    result_bary_wp = wp.zeros(num_query_points, dtype=wp.vec2)

    wp.launch(
        kernel=get_closest_nodes_and_faces,
        dim=num_query_points,
        inputs=[query_points_wp, mesh.id, num_query_points, result_face_wp, result_node_wp, result_bary_wp, max_distance],
    )

    result_face_out = wp.to_torch(result_face_wp)
    result_node_out = wp.to_torch(result_node_wp)
    result_bary_out = wp.to_torch(result_bary_wp)

    indices_from = torch.arange(num_query_points, device=query_points.device)
    valid_mask = result_face_out != -1

    result_face_out = result_face_out[valid_mask]
    result_node_out = result_node_out[valid_mask]
    result_bary_out = result_bary_out[valid_mask]
    indices_from = indices_from[valid_mask]


    return indices_from, result_node_out, result_face_out, result_bary_out


@wp.func
def check_and_add_point_triangle(point: wp.vec3, v0: wp.vec3, v1: wp.vec3, v2: wp.vec3,
                                 max_distance: float, mesh_id: wp.uint64,
                                 vid: wp.int32, fid: wp.int32,
                                result_node_pairs: wp.array(dtype=wp.vec2i),
                                result_face: wp.array(dtype=wp.int32),
                                result_bary: wp.array(dtype=wp.vec3),
                                counter: wp.array(dtype=wp.int32),
                                max_index: wp.int32):
    if is_node_in_face(mesh_id, fid, vid):
        return


    barycoords = get_barycoords(point, v0, v1, v2)


    # if vid == 8 and fid == 211:
    #     printf("tid: %d, f2: %d\n", vid, fid)
    #     print(barycoords)

    for i in range(3):
        # TODO: add gap
        if barycoords[i] < 0 or barycoords[i] > 1.:
            return

    distance = get_point_plain_distance(point, v0, v1, v2)

    # if vid == 8 and fid == 211:
    #     printf("distance: %f\n", distance)

    if distance <= max_distance:
        index_to_add = wp.atomic_add(counter, 0, 1)
        if index_to_add > max_index:
            print('Number of node-face pairs exceeded max_index')
        else:
            result_face[index_to_add] = fid
            result_bary[index_to_add] = barycoords

            node = choose_closest_node(mesh_id, fid, point)
            result_node_pairs[index_to_add] = wp.vec2i(vid, wp.mesh_get_index(mesh_id, fid * 3 + node))





@wp.kernel
def get_proximity_self(
        query_points: wp.array(dtype=wp.vec3),
        query_ids: wp.array(dtype=wp.int32),
        mesh_id: wp.uint64,
        num_query_points: int,
        result_node_pairs: wp.array(dtype=wp.vec2i),
        result_face: wp.array(dtype=wp.int32),
        result_bary: wp.array(dtype=wp.vec3),
        counter: wp.array(dtype=wp.int32),
        max_distance: float,
        max_index: wp.int32
):


    tid = wp.tid()
    if tid >= num_query_points:
        return

    point = query_points[tid]

    lower = vs_add(point, -max_distance)
    upper = vs_add(point, max_distance)

    vid = query_ids[tid]

    query = wp.mesh_query_aabb(mesh_id, lower, upper)
    for f2 in query:
        u0 = wp.mesh_get_point(mesh_id, f2 * 3)
        u1 = wp.mesh_get_point(mesh_id, f2 * 3 + 1)
        u2 = wp.mesh_get_point(mesh_id, f2 * 3 + 2)


        check_and_add_point_triangle(point, u0, u1, u2, max_distance, mesh_id, vid, f2,
                                     result_node_pairs, result_face, result_bary,
                                     counter, max_index)
        

        # check_and_add_point_triangle3(point, u0, u1, u2, max_distance, mesh_id, vid, f2,
        #                              result_node_pairs, result_face, result_bary,
        #                              counter, max_index)


@wp.kernel
def set_points_to_mesh(
    mesh_id: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    num_points: int
):
    tid = wp.tid()
    pid = tid % num_points

    mesh = wp.mesh_get(mesh_id)
    mesh.points[pid] = points[pid]

class WarpCacheChecker:
    def __init__(self):
        self.cache = {}

    def add_mesh(self, mesh_name, mesh_verts, mesh_faces):
        key = mesh_name
        if key in self.cache:
            return
        else:
            mesh_verts_wp = wp.from_torch(mesh_verts.float().contiguous(), dtype=wp.vec3)
            faces_wp = wp.from_torch(mesh_faces.reshape(-1).contiguous().int())
            mesh = wp.Mesh(
                points=mesh_verts_wp,
                indices=faces_wp)
            self.cache[key] = mesh

    def get(self, mesh_name, mesh_verts, mesh_faces):
        key = mesh_name
        if key in self.cache:
            return self.cache[key]
        else:
            mesh_verts_wp = wp.from_torch(mesh_verts.float().contiguous(), dtype=wp.vec3)
            faces_wp = wp.from_torch(mesh_faces.reshape(-1).contiguous().int())
            mesh = wp.Mesh(
                points=mesh_verts_wp,
                indices=faces_wp)
            self.cache[key] = mesh
            return mesh

    def get_closest_nodes_and_faces(self, query_points, mesh_verts, mesh_faces, max_distance, mesh_name=None):
        faces_wp = wp.from_torch(mesh_faces.reshape(-1).int())
        mesh_verts_wp = wp.from_torch(mesh_verts.float(), dtype=wp.vec3)
        query_points_wp = wp.from_torch(query_points.float(), dtype=wp.vec3)
        n_verts = mesh_verts.shape[0]

        if mesh_name is None:
            mesh = wp.Mesh(
                points=mesh_verts_wp,
                indices=faces_wp,
            )
        else:
            mesh = self.get(mesh_name, mesh_verts, mesh_faces)
            wp.launch(
                kernel=set_points_to_mesh,
                dim=n_verts,
                inputs=[mesh.id, mesh_verts_wp, n_verts],
            )
            mesh.refit()

        num_query_points = query_points.shape[0]
        result_face_pt = torch.ones(num_query_points, dtype=torch.int32, device=query_points.device) * -1
        result_face_wp = wp.from_torch(result_face_pt.clone(), dtype=wp.int32)
        result_node_wp = wp.from_torch(result_face_pt.clone(), dtype=wp.int32)
        result_bary_wp = wp.zeros(num_query_points, dtype=wp.vec2)

        wp.launch(
            kernel=get_closest_nodes_and_faces,
            dim=num_query_points,
            inputs=[query_points_wp, mesh.id, num_query_points, result_face_wp, result_node_wp, result_bary_wp,
                    max_distance],
        )

        result_face_out = wp.to_torch(result_face_wp)
        result_node_out = wp.to_torch(result_node_wp)
        result_bary_out = wp.to_torch(result_bary_wp)

        indices_from = torch.arange(num_query_points, device=query_points.device)
        valid_mask = result_face_out != -1

        result_face_out = result_face_out[valid_mask]
        result_node_out = result_node_out[valid_mask]
        result_bary_out = result_bary_out[valid_mask]
        indices_from = indices_from[valid_mask]

        return indices_from, result_node_out, result_face_out, result_bary_out

    def get_proximity_self(self, query_points, mesh_verts, mesh_faces, max_distance, pairs_pre_point=32, query_ids=None, mesh_name=None):
        """
        pytorch wrapper for get_prowimity_self
        """
        faces_wp = wp.from_torch(mesh_faces.reshape(-1).contiguous().int())
        mesh_verts_wp = wp.from_torch(mesh_verts.float().contiguous(), dtype=wp.vec3)
        query_points_wp = wp.from_torch(query_points.float().contiguous(), dtype=wp.vec3)
        n_verts = mesh_verts.shape[0]

        # print('mesh_verts', mesh_verts.shape)
        # print('n_verts', n_verts)

        # mesh = wp.Mesh(
        #     points=mesh_verts_wp,
        #     indices=faces_wp,
        # )

        if mesh_name is None:
            mesh = wp.Mesh(
                points=mesh_verts_wp,
                indices=faces_wp,
            )
        else:
            mesh = self.get(mesh_name, mesh_verts, mesh_faces)
            wp.launch(
                kernel=set_points_to_mesh,
                dim=n_verts,
                inputs=[mesh.id, mesh_verts_wp, n_verts],
            )
            mesh.refit()

        num_query_points = query_points.shape[0]
        result_face_pt = torch.ones(num_query_points * pairs_pre_point, dtype=torch.int32, device=query_points.device) * -1
        result_face_wp = wp.from_torch(result_face_pt.clone(), dtype=wp.int32)

        result_node_pairs_pt = torch.ones(num_query_points * pairs_pre_point, 2, dtype=torch.int32, device=query_points.device) * -1
        result_node_pairs_wp = wp.from_torch(result_node_pairs_pt.clone(), dtype=wp.vec2i)

        result_bary_pt = torch.ones(num_query_points * pairs_pre_point, 3, dtype=torch.float32, device=query_points.device) * -1
        result_bary_wp = wp.from_torch(result_bary_pt.clone(), dtype=wp.vec3)

        counter_pt = torch.zeros(1, dtype=torch.int32, device=query_points.device)
        counter_wp = wp.from_torch(counter_pt.clone(), dtype=wp.int32)

        if query_ids is None:
            query_ids = torch.arange(num_query_points, device=query_points.device)

        query_ids = wp.from_torch(query_ids.int().contiguous(), dtype=wp.int32)

        max_index = num_query_points * pairs_pre_point
        wp.launch(
            kernel=get_proximity_self,
            dim=num_query_points,
            inputs=[query_points_wp, query_ids, mesh.id, num_query_points, result_node_pairs_wp, result_face_wp, result_bary_wp, counter_wp, max_distance, max_index],
        )

        result_face_out = wp.to_torch(result_face_wp)
        result_node_pairs_out = wp.to_torch(result_node_pairs_wp)
        result_bary_out = wp.to_torch(result_bary_wp)

        counter_out = wp.to_torch(counter_wp)

        mask = result_face_out != -1
        result_face_out = result_face_out[mask]
        result_node_pairs_out = result_node_pairs_out[mask]
        result_bary_out = result_bary_out[mask]

        return result_face_out, result_node_pairs_out, result_bary_out, counter_out





def get_proximity_self_pt_dummmy(query_points, mesh_verts, mesh_faces, max_distance, pairs_pre_point=32, query_ids=None):
    """
    pytorch wrapper for get_prowimity_self
    """
    faces_wp = wp.from_torch(mesh_faces.reshape(-1).contiguous().int())
    mesh_verts_wp = wp.from_torch(mesh_verts.float().contiguous(), dtype=wp.vec3)
    query_points_wp = wp.from_torch(query_points.float().contiguous(), dtype=wp.vec3)


    mesh = wp.Mesh(
        points=mesh_verts_wp,
        indices=faces_wp,
    )

    num_query_points = query_points.shape[0]
    result_face_pt = torch.ones(num_query_points * pairs_pre_point, dtype=torch.int32, device=query_points.device) * -1
    result_face_wp = wp.from_torch(result_face_pt.clone(), dtype=wp.int32)

    result_node_pairs_pt = torch.ones(num_query_points * pairs_pre_point, 2, dtype=torch.int32, device=query_points.device) * -1
    result_node_pairs_wp = wp.from_torch(result_node_pairs_pt.clone(), dtype=wp.vec2i)

    result_bary_pt = torch.ones(num_query_points * pairs_pre_point, 3, dtype=torch.float32, device=query_points.device) * -1
    result_bary_wp = wp.from_torch(result_bary_pt.clone(), dtype=wp.vec3)

    counter_pt = torch.zeros(1, dtype=torch.int32, device=query_points.device)
    counter_wp = wp.from_torch(counter_pt.clone(), dtype=wp.int32)

    if query_ids is None:
        query_ids = torch.arange(num_query_points, device=query_points.device)

    query_ids = wp.from_torch(query_ids.int().contiguous(), dtype=wp.int32)

    max_index = num_query_points * pairs_pre_point
    wp.launch(
        kernel=get_proximity_self,
        dim=num_query_points,
        inputs=[query_points_wp, query_ids, mesh.id, num_query_points,
                result_node_pairs_wp, result_face_wp, result_bary_wp,
                counter_wp, max_distance, max_index],
    )

    result_face_out = wp.to_torch(result_face_wp)
    result_node_pairs_out = wp.to_torch(result_node_pairs_wp)
    result_bary_out = wp.to_torch(result_bary_wp)

    counter_out = wp.to_torch(counter_wp)


    # print('max_distance', max_distance)
    # print('counter_out', counter_out)
    # assert False


    mask = result_face_out != -1
    result_face_out = result_face_out[mask]
    result_node_pairs_out = result_node_pairs_out[mask]
    result_bary_out = result_bary_out[mask]

    return result_face_out, result_node_pairs_out, result_bary_out, counter_out

