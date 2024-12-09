import numpy as np
from transforms3d.quaternions import qinverse, qconjugate, qmult, qnorm, quat2mat, mat2quat, quat2axangle, axangle2quat, nearly_equivalent
from transforms3d.euler import euler2quat, quat2euler, euler2mat, mat2euler
from scipy.spatial.transform import Rotation, Slerp


# Applying a delta quaternion in a global context (multiply to the left):
#         q2 = delta_q_global * q1
#         delta_q_global = q_2 * inv(q_1)
# in a reference frame q_ref, where q1_prime = inv(q_ref) * q1, q2_prime = inv(q_ref) * q2
#         q2_prime = delta_q_prime_global * q1_prime
#         delta_q_prime_global = q2_prime * inv(q1_prime)
#                              = (inv(q_ref) * q2) * inv(inv(q_ref) * q1)
#                              = inv(q_ref) * q2 * inv(q1) * q_ref
#                              = inv(q_ref) * delta_q_global * q_ref
# when q_ref = q_1
#         delta_q_prime_global = inv(q_1) * delta_q_global * q_1
#                              = inv(q_1) * (q_2 * inv(q_1)) * q_1
#                              = inv(q_1) * q_2
#                              = delta_q_local
# Here, the delta quaternion is applied globally, which means the rotation it represents
# is done in the world coordinate system. This is typically used when the rotation should
# not depend on the object's current orientation but rather on a fixed external frame of reference.
# Example: A world-aligned wind pushing a weathervane to point north regardless of its current orientation.

# Applying a delta quaternion in a local context (multiply to the right):
#         q2 = q1 * delta_q_local
#         delta_q_local = inv(q_1) * q_2
# in a reference frame q_ref, where q1_prime = inv(q_ref) * q1, q2_prime = inv(q_ref) * q2
#         q2_prime = q1_prime * delta_q_prime_local
#         delta_q_prime_local = inv(q1_prime) * q2_prime
#                             = inv(inv(q_ref) * q1) * (inv(q_ref) * q2)
#                             = inv(q1) * q_ref * inv(q_ref) * q2
#                             = delta_q_local
# In this case, the delta quaternion is applied in the local coordinate system of the object.
# This means the rotation is relative to the object's current orientation. This approach is commonly
# used in scenarios like robotics or character animation, where each incremental movement is based
# on the current orientation of the robot or character.
# Example: A character turning its head an additional 30 degrees to the left relative to its current pose.


def delta_quaternion_to_angluar_velocity(delta_q, dt):
    # custom solution
    # theta = 2 * np.arctan2(np.linalg.norm(delta_q[1:]), delta_q[0])
    # v = delta_q[1:] / np.linalg.norm(delta_q[1:]) if np.linalg.norm(
    #     delta_q[1:]) != 0 else np.array([0, 0, 1])
    # official solution
    v, theta = quat2axangle(delta_q)
    omega = theta / dt * v  # Angular velocity vector
    return omega


def angluar_velocity_to_delta_quaternion(omega, dt):
    norm_omega = np.linalg.norm(omega)
    if norm_omega == 0:
        delta_q = np.array([1.0, 0.0, 0.0, 0.0])  # No rotation
        return delta_q
    theta = norm_omega * dt  # Angle of rotation
    u = omega / norm_omega  # Normalized axis of rotation
    # custom solution
    # w = np.cos(theta / 2)
    # x, y, z = u * np.sin(theta / 2)
    # delta_q = np.array([w, x, y, z])
    # official solution
    delta_q = axangle2quat(u, theta)
    return delta_q


def quaternions_to_angular_velocity(q1, q2, dt, option='quaternion'):
    """
    Converts the rotation between two quaternions into angular velocity.

    Parameters:
        q1 (array): The initial quaternion representing the camera pose.
        q2 (array): The final quaternion representing the camera pose.
        dt (float): The time difference between q1 and q2.

    Returns:
        omega_global (array): The angular velocity vector.

    References:
        - https://stackoverflow.com/questions/56790186/how-to-compute-angular-velocity-using-numpy-quaternion
        - https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation

    Notes:
        - The angular velocity is computed based on delta_q, the operation that brings q1 to q2.
    """
    # R2 = R @ R1
    if option == 'matrix':
        R1 = quat2mat(q1)
        R2 = quat2mat(q2)
        R = R2 @ R1.T
        omega_global = np.array(quat2euler(mat2quat(R))) / dt
    elif option == 'quaternion':
        delta_q_global = qmult(q2, qinverse(q1))
        delta_q_global /= qnorm(delta_q_global)
        omega_global = delta_quaternion_to_angluar_velocity(delta_q_global, dt)
    else:
        raise ValueError(f'Invalid option: {option}')
    return omega_global


def add_angular_velocity_to_quaternion(q1, omega_global, dt, option='quaternion'):
    """
    Adds angular velocity to a quaternion.

    Parameters:
        q1 (array): The initial quaternion [w, x, y, z].
        omega_local (array): The angular velocity vector [wx, wy, wz] for delta_q.
        dt (float): The time step.

    Returns:
        q2 (array): The updated quaternion [w, x, y, z].

    Notes:
        - The angular velocity is computed based on delta_q, the operation that brings q1 to q2.
    """
    # R2 = delta_R @ R1
    if option == 'matrix':
        R1 = quat2mat(q1)
        delta_R = euler2mat(*(omega_global * dt))
        q2 = mat2quat(delta_R @ R1)
    elif option == 'quaternion':
        delta_q_global = angluar_velocity_to_delta_quaternion(omega_global, dt)
        q2 = qmult(delta_q_global, q1)
    else:
        raise ValueError(f'Invalid option: {option}')
    return q2


def convert_to_local_frame(tvec_ref, qvec_ref, tvec_global=None, qvec_global=None, v_global=None, omega_global=None, option='quaternion'):
    """
    Convert position, orientation, velocity, and angular velocity from the global coordinate system
    (frame) to the local coordinate system (frame) defined by a reference position and orientation.

    Parameters:
    - tvec_ref: Reference position vector in the global frame [x, y, z].
    - qvec_ref: Reference orientation quaternion in the global frame [qw, qx, qy, qz].
    - tvec_global: Global position vector [x, y, z].
    - qvec_global: Global orientation quaternion [qw, qx, qy, qz].
    - v_global: Global velocity vector [vx, vy, vz].
    - omega_global: Angular velocity vector [wx, wy, wz].

    Returns:
    - tvec_local: Position in the local frame.
    - qvec_local: Orientation in the local frame.
    - v_local: Velocity in the local frame.
    - omega_local: Angular velocity in the local frame.
    """
    # Convert reference quaternion to rotation matrix
    R = quat2mat(qvec_ref)

    # Translate global position to relative position and rotate to get local position
    tvec_local = R.T @ (tvec_global -
                        tvec_ref) if tvec_global is not None else None

    # Calculate the relative orientation quaternion
    if qvec_global is not None:
        if option == 'matrix':
            R_local = R.T @ quat2mat(qvec_global)
            qvec_local = mat2quat(R_local)
        elif option == 'quaternion':
            qvec_local = qmult(qinverse(qvec_ref), qvec_global)
    else:
        qvec_local = None

    # Rotate global velocity to get local velocity
    v_local = R.T @ v_global if v_global is not None else None

    # Rotate global angular velocity to get local angular velocity
    if omega_global is not None:
        if option == 'matrix':
            # R1_global = R @ R1_local
            # R2_global = R @ R2_local
            # R2_local = delta_R_local @ R1_local
            # R2_global = delta_R_global @ R1_global
            # R @ R2_local = delta_R_local @ R @ R1_local
            # R2_local = (R.T @ delta_R_global @ R) @ R1_local
            delta_R_global = euler2mat(*(omega_global))
            delta_R_local = R.T @ delta_R_global @ R
            omega_local = mat2euler(delta_R_local)
        elif option == 'quaternion':
            omega_local = R.T @ omega_global
        else:
            raise ValueError(f'Invalid option: {option}')
    else:
        omega_local = None

    return tvec_local, qvec_local, v_local, omega_local


def convert_to_global_frame(tvec_ref, qvec_ref, tvec_local=None, qvec_local=None, v_local=None, omega_local=None, option='quaternion'):
    """
    Convert position, orientation, velocity, and angular velocity from the local coordinate system
    (frame) to the global coordinate system (frame) defined by a reference position and orientation.

    Parameters:
    - tvec_ref: Reference position vector in the global frame [x, y, z].
    - qvec_ref: Reference orientation quaternion in the global frame [qw, qx, qy, qz].
    - tvec_local: Local position vector [x, y, z].
    - qvec_local: Local orientation quaternion [qw, qx, qy, qz].
    - v_local: Local velocity vector [vx, vy, vz].
    - omega_local: Angular velocity vector [wx, wy, wz].

    Returns:
    - tvec_global: Position in the global frame.
    - qvec_global: Orientation in the global frame.
    - v_global: Velocity in the global frame.
    - omega_global: Angular velocity in the global frame.
    """
    # Convert reference quaternion to rotation matrix
    R = quat2mat(qvec_ref)

    # Rotate local position to global frame and then translate
    tvec_global = R @ tvec_local + tvec_ref if tvec_local is not None else None

    # Calculate the global orientation quaternion
    if qvec_local is not None:
        if option == 'matrix':
            R_global = R @ quat2mat(qvec_local)
            qvec_global = mat2quat(R_global)
        elif option == 'quaternion':
            qvec_global = qmult(qvec_ref, qvec_local)
    else:
        qvec_global = None

    # Rotate local velocity to get global velocity
    v_global = R @ v_local if v_local is not None else None

    # Rotate local angular velocity to get global angular velocity
    if omega_local is not None:
        if option == 'matrix':
            # R1_local = R.T @ R1_global
            # R2_local = R.T @ R2_global
            # R2_local = delta_R_local @ R1_local
            # R2_global = delta_R_global @ R1_global
            # R.T @ R1_global = delta_R_local @ R.T @ R1_global
            # R1_global = (R @ delta_R_local @ R.T) @ R1_global
            delta_R_local = euler2mat(*(omega_local))
            delta_R_global = R @ delta_R_local @ R.T
            omega_global = mat2euler(delta_R_global)
        elif option == 'quaternion':
            omega_global = R @ omega_local
    else:
        omega_global = None

    return tvec_global, qvec_global, v_global, omega_global


def interpolate_tvecs(tvecs, times, new_times):
    interpolated_tvecs = []
    for t in new_times:
        i1 = np.searchsorted(times, t) - 1
        i2 = min(i1 + 1, len(times) - 1)
        i1 = max(0, i1)
        t1, t2 = times[i1], times[i2]
        tvec1, tvec2 = tvecs[i1], tvecs[i2]
        tvec = tvec1 + (t - t1) / (t2 - t1 + 1e-8) * (tvec2 - tvec1)
        interpolated_tvecs.append(tvec)
    return np.array(interpolated_tvecs)


def interpolate_eulers(rots, times, new_times):
    key_rots = Rotation.from_euler('xyz', rots)
    slerp = Slerp(times, key_rots)
    interpolated_rots = slerp(new_times).as_euler('xyz')
    return interpolated_rots


def horizontal_flip(tvec, qvec, v=None, omega=None):
    """
    Equavalent operation on the position, orientation, velocity, and angular velocity to a horizontal flip to the image.

    Parameters:
    - tvec: Position vector [x, y, z].
    - qvec: Orientation quaternion [qw, qx, qy, qz].
    - v: Velocity vector [vx, vy, vz].
    - omega: Angular velocity vector [wx, wy, wz].

    Returns:
    - tvec_flipped: Flipped position vector.
    - qvec_flipped: Flipped orientation quaternion.
    - v_flipped: Flipped velocity vector.
    - omega_flipped: Flipped angular velocity vector.
    """
    # Flip the x-coordinate of the position vector
    tvec_flipped = np.array([-tvec[0], tvec[1], tvec[2]])

    # Flip the sign of the y and z components of the orientation quaternion
    qvec_flipped = np.array([qvec[0], qvec[1], -qvec[2], -qvec[3]])

    # Flip the sign of the x component of the velocity vector
    if v is not None:
        v_flipped = np.array([-v[0], v[1], v[2]])
    else:
        v_flipped = None

    # Flip the sign of the y and z components of the angular velocity vector
    if omega is not None:
        omega_flipped = np.array([omega[0], -omega[1], -omega[2]])
    else:
        omega_flipped = None

    return tvec_flipped, qvec_flipped, v_flipped, omega_flipped


def main():
    from transforms3d.quaternions import qinverse, qconjugate, qmult, qnorm, quat2mat, mat2quat, quat2axangle, axangle2quat, nearly_equivalent
    from transforms3d.euler import euler2quat, quat2euler, euler2mat, mat2euler

    # Example usage
    q1 = np.array([0.0, 0.0, 1.0, 0.0])  # Quaternion at time t1
    q2 = np.array([0.0, 0.0, 0.0, 1.0])  # Quaternion at time t2, for example
    dt = 1  # Time at q1

    qvecs = np.array([[9.99219848e-01,  1.31254768e-02, -3.50585785e-02,
                     1.25822934e-02],
                      [9.99068925e-01,  1.49084515e-02, -3.93790465e-02,
                     9.39747139e-03],
                      [9.98888877e-01,  1.72810791e-02, -4.34385416e-02,
                     5.95562585e-03],
                      [9.98645212e-01,  1.88819139e-02, -4.84316454e-02,
                       2.36431896e-03],
                      [9.98347449e-01,  2.04531995e-02, -5.36964746e-02,
                       -8.51922491e-04]])
    q1 = qvecs[1]
    q2 = qvecs[2]

    # convert q1 and q2 to rotation matrix
    R1 = quat2mat(q1)
    R2 = quat2mat(q2)
    # convert the rotation matrix to euler angles
    euler1 = quat2euler(q1)
    euler2 = quat2euler(q2)
    euler1_ = mat2euler(R1)
    euler2_ = mat2euler(R2)

    # Calculate delta quaternion in global frame
    # q2 = q1 * delta_q
    delta_q_global = qmult(q2, qinverse(q1))
    delta_R_global = R2 @ R1.T

    # Calculate delta quaternion in local frame
    # q2 = delta_q * q1
    delta_q_local = qmult(qinverse(q1), q2)
    delta_R_local = R1.T @ R2

    q_ref = q1
    q1_prime = qmult(qinverse(q_ref), q1)
    q2_prime = qmult(qinverse(q_ref), q2)
    R_ref = R1
    R1_prime = R_ref.T @ R1
    R2_prime = R_ref.T @ R2

    # Calculate delta quaternion in global frame
    # q2_prime = q1_prime * delta_q
    # delta_q_prime_global = q2_prime * inv(q1_prime)
    #                      = (inv(q_ref) * q2) * inv(inv(q_ref) * q1)
    #                      = inv(q_ref) * q2 * inv(q1) * q_ref
    #                      = inv(q_ref) * delta_q_global * q_ref
    delta_q_prime_global = qmult(q2_prime, qinverse(q1_prime))
    delta_q_prime_global_ = qmult(
        qmult(qinverse(q_ref), delta_q_global), q_ref)

    # Calculate delta quaternion in local frame
    # q2_prime = delta_q * q1_prime
    # delta_q_prime_local = inv(q1_prime) * q2_prime
    #                     = inv(inv(q_ref) * q1) * (inv(q_ref) * q2)
    #                     = inv(q1) * q_ref * inv(q_ref) * q2
    #                     = delta_q_local
    delta_q_prime_local = qmult(qinverse(q1_prime), q2_prime)
    delta_q_prime_local_ = delta_q_local

    def calculate_difference(q1, q2):
        """Calculate the difference in location and rotation between two poses."""
        # This matrix R will transform a point in the world coordinate system into the camera coordinate system.
        # s * [x,y,z,1] (camear coord) = [R,t] * [X,Y,Z,1] (world coord)
        R1 = quat2mat(q1)
        R2 = quat2mat(q2)

        # Compute the difference in transformation
        R_diff = R2 @ R1.T

        # Extract rotation difference
        rotation_diff_quat = mat2quat(R_diff)

        return rotation_diff_quat

    # Calculate the relative orientation quaternion
    rotation_diff_quat = calculate_difference(q1, q2)

    omega_global = quaternions_to_angular_velocity(q1, q2, dt)
    q2_ = add_angular_velocity_to_quaternion(q1, omega_global, dt)
    print(np.abs(q2 - q2_).max(), nearly_equivalent(q2, q2_))

    # re-run under relative coordinates
    # from convert_to_global_frame()
    qvec_ref = qvecs[0]
    qvecs_rlt = np.zeros_like(qvecs)

    R = quat2mat(qvec_ref)

    # Calculate the relative orientation quaternion
    for i in range(len(qvecs)):
        qvecs_rlt[i] = qmult(qinverse(qvec_ref), qvecs[i])
    q1_rlt = qvecs_rlt[1]
    q2_rlt = qvecs_rlt[2]

    # option 1:
    _, _, _, omega_rlt1 = convert_to_local_frame(np.zeros(3), qvec_ref,
                                                 None, None, None, omega_global)
    q2_rlt1 = add_angular_velocity_to_quaternion(q1_rlt, omega_rlt1, dt)
    print(np.abs(q2_rlt - q2_rlt1).max(), nearly_equivalent(q2_rlt, q2_rlt1))

    # option 2: calculate the relative angular velocity from relative quaternions
    omega_rlt2 = quaternions_to_angular_velocity(q1_rlt, q2_rlt, dt)
    q2_rlt2 = add_angular_velocity_to_quaternion(q1_rlt, omega_rlt2, dt)
    print(np.abs(q2_rlt - q2_rlt2).max(), nearly_equivalent(q2_rlt, q2_rlt2))
    print()
    pass


if __name__ == '__main__':
    main()
