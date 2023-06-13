import torch
from smplx.lbs import blend_shapes, vertices2joints, batch_rodrigues, batch_rigid_transform
from smplx.utils import Tensor



def get_shaped_joints(
        betas: Tensor,
        v_template: Tensor,
        shapedirs: Tensor,
        J_regressor: Tensor,

):
    # Add shape contribution
    v_shaped = v_template + blend_shapes(betas, shapedirs)

    # Get the joints
    # NxJx3 array
    J = vertices2joints(J_regressor, v_shaped)

    return J


def get_transformed_joints(
        betas: Tensor,
        pose: Tensor,
        v_template: Tensor,
        shapedirs: Tensor,
        J_regressor: Tensor,
        parents: Tensor,
        pose2rot: bool = True,

):
    batch_size = max(betas.shape[0], pose.shape[0])
    device, dtype = betas.device, betas.dtype
    J = get_shaped_joints(betas, v_template, shapedirs, J_regressor)
    # 3. Add pose blend shapes
    # N x J x 3 x 3
    if pose2rot:
        rot_mats = batch_rodrigues(pose.view(-1, 3)).view(
            [batch_size, -1, 3, 3])
    else:
        rot_mats = pose.view(batch_size, -1, 3, 3)

    J_transformed, A = batch_rigid_transform(rot_mats, J, parents, dtype=dtype)
    return J_transformed, A


def pose_garment(
        betas: Tensor,
        pose: Tensor,
        v_template: Tensor,
        shapedirs: Tensor,
        posedirs: Tensor,
        lbs_weights: Tensor,
        J_transformed: Tensor,
        joint_transforms: Tensor,
        pose2rot: bool = True
):
    """
    Generate the garment geometry from the given SMPL pose and shape parameters.
    :param betas: [Nx10] shape parameters
    :param pose: [Nx72] pose parameters
    :param v_template: [Vx3] garment template geometry
    :param shapedirs: [Vx3x10] shape blend shapes
    :param posedirs: [69*3 x V*3] pose blend shapes
    :param lbs_weights: [Vx24] linear blend skinning weights
    :param J_transformed: [Nx24x3] joint locations after applying pose blend shapes
    :param joint_transforms: [Nx24x4x4] joint transforms
    :param pose2rot: whether to convert the pose parameters to rotation matrices

    :return: verts: [NxVx3] garment vertices in the given pose and shape
    :return: J_transformed: [Nx24x3] joint locations in the given pose and shape
    """
    batch_size = max(betas.shape[0], pose.shape[0])
    device, dtype = betas.device, betas.dtype

    # Add shape contribution
    v_shaped = v_template + blend_shapes(betas, shapedirs)

    A = joint_transforms

    ident = torch.eye(3, dtype=dtype, device=device)
    if pose2rot:
        rot_mats = batch_rodrigues(pose.view(-1, 3)).view(
            [batch_size, -1, 3, 3])
        pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])
        # (N x P) x (P, V * 3) -> N x V x 3
        pose_offsets = torch.matmul(
            pose_feature, posedirs).view(batch_size, -1, 3)
    else:
        pose_feature = pose[:, 1:].view(batch_size, -1, 3, 3) - ident

        pose_offsets = torch.matmul(pose_feature.view(batch_size, -1),
                                    posedirs).view(batch_size, -1, 3)

    v_posed = pose_offsets + v_shaped

    # 5. Do skinning:
    # W is N x V x (J + 1)
    W = lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
    # (N x V x (J + 1)) x (N x (J + 1) x 16)
    num_joints = J_transformed.shape[1]
    T = torch.matmul(W, A.view(batch_size, num_joints, 16)) \
        .view(batch_size, -1, 4, 4)

    homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1],
                               dtype=dtype, device=device)
    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
    v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

    verts = v_homo[:, :, :3, 0]

    return verts, J_transformed
