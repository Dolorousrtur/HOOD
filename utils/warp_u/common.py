import warp as wp



@wp.func
def vs_add(a: wp.vec3, b: float):
    return wp.vec3(a[0]+b, a[1]+b, a[2]+b)

@wp.func
def get_barycoords(point: wp.vec3, v0: wp.vec3, v1: wp.vec3, v2: wp.vec3):
    # print('triangles', triangles.shape)
    # print('points', points.shape)


    e1 = v0 - v1
    e2 = v2 - v1
    ep = point - v1

    d00 = wp.dot(e1, e1)
    d01 = wp.dot(e1, e2)
    d11 = wp.dot(e2, e2)
    d20 = wp.dot(ep, e1)  # ?
    d21 = wp.dot(ep, e2)  # ?
    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1. - v - w

    barycoords =  wp.vec3(v, u, w) # TODO: compute only u and v
    return barycoords



@wp.func
def get_normal_vec( v0: wp.vec3, v1: wp.vec3, v2: wp.vec3):
    e1 = v1 - v0
    e2 = v2 - v0
    n = wp.cross(e1, e2)
    n = wp.normalize(n)
    return n

@wp.func
def get_point_plain_distance(point: wp.vec3, v0: wp.vec3, v1: wp.vec3, v2: wp.vec3):
    n = get_normal_vec(v0, v1, v2)
    p2p = point - v0

    p2p_n = wp.dot(p2p, n)
    distance = wp.abs(p2p_n)
    return distance



@wp.func
def is_node_in_face(mesh_id: wp.uint64, fid: wp.int32, vid: wp.int32):
    for i in range(3):
        if wp.mesh_get_index(mesh_id, fid * 3+i) == vid:
            return True

    return False


@wp.func
def get_point_face_distance(point: wp.vec3, v0: wp.vec3, v1: wp.vec3, v2: wp.vec3):
    barycoords = get_barycoords(point, v0, v1, v2)

    e1 = v0 - v1
    e2 = v2 - v1
    ep = point - v1


    u = barycoords[0]
    v = barycoords[1]
    w = barycoords[2]

    v_valid = v > 0. and v < 1.
    w_valid = w > 0. and w < 1.
    u_valid = u > 0. and u < 1.
    all_valid = v_valid and w_valid and u_valid

    if not all_valid:
        if u <= 0:
            if v <= 0:
                return wp.length(v2 - point)
            elif w <= 0:
                return wp.length(v1 - point)
            else:
                w = wp.dot(ep, e2) / wp.dot(e2, e2)
                w = wp.clamp(w, 0., 1.)
                v = 1. - w

                pt = v1 * v + v2 * w
                return wp.length(pt - point)
        elif v <= 0:
            if w <= 0:
                return wp.length(v0 - point)
            else:
                edge_t = v2 - v0
                edge_p = point - v0

                w = wp.dot(edge_p, edge_t) / wp.dot(edge_t, edge_t)
                w = wp.clamp(w, 0., 1.)
                u = 1. - w

                pt = v0 * u + v2 * w
                return wp.length(pt - point)
        else: # w <= 0
            u = wp.dot(ep, e1) / wp.dot(e1, e1)
            u = wp.clamp(u, 0., 1.)
            v = 1. - u

            pt = v0 * u + v1 * v
            return wp.length(pt - point)
    else:
        return get_point_plain_distance(point, v0, v1, v2)