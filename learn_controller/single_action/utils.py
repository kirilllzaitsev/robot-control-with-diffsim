import numpy as np
import pybullet as p


def print_pb_obj_state(obj_id):
    print("#" * 10)
    cubePos, cubeOrn = get_obj_state(obj_id)
    print("cubePos=", cubePos)
    print("cubeOrn=", cubeOrn)
    print("#" * 10)


def get_obj_state(obj_id):
    cubePos, cubeOrn = p.getBasePositionAndOrientation(obj_id)
    return cubePos, cubeOrn


def render(robot):
    pos, rot, _, _, _, _ = p.getLinkState(
        robot.robot_id, linkIndex=robot.end_eff_idx, computeForwardKinematics=True
    )
    rot_matrix = p.getMatrixFromQuaternion(rot)
    rot_matrix = np.array(rot_matrix).reshape(3, 3)

    # camera params
    height = 640
    width = 480
    fx, fy = 596.6278076171875, 596.6278076171875
    cx, cy = 311.98663330078125, 236.76170349121094
    near, far = 0.1, 10

    camera_vector = rot_matrix.dot((0, 0, 1))
    up_vector = rot_matrix.dot((0, -1, 0))

    camera_eye_pos = np.array(pos)
    camera_target_position = camera_eye_pos + 0.2 * camera_vector

    view_matrix = p.computeViewMatrix(camera_eye_pos, camera_target_position, up_vector)

    proj_matrix = (
        2.0 * fx / width,
        0.0,
        0.0,
        0.0,
        0.0,
        2.0 * fy / height,
        0.0,
        0.0,
        1.0 - 2.0 * cx / width,
        2.0 * cy / height - 1.0,
        (far + near) / (near - far),
        -1.0,
        0.0,
        0.0,
        2.0 * far * near / (near - far),
        0.0,
    )

    p.getCameraImage(
        width=width,
        height=height,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
    )  # renderer=self._p.ER_TINY_RENDERER)
