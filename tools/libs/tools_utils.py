import numpy as np

from scipy.spatial.transform import Rotation as R

from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import TransformStamped


def xyzq_to_matrix(translation, quaternion):
    """
    Convert translation and quaternion to a 4x4 transformation matrix.

    Arguments:
        translation (list): List of translation values [x, y, z].
        quaternion (list): List of quaternion values [x, y, z, w].

    Returns:
        numpy.ndarray:
            4x4 transformation matrix representing the translation and rotation.
    """
    x, y, z = translation

    # Create a rotation matrix from the quaternion
    rotation_matrix = R.from_quat(quaternion).as_matrix()

    # Create a transformation matrix
    matrix = np.eye(4)
    matrix[:3, :3] = rotation_matrix
    matrix[:3, 3] = [x, y, z]

    return matrix


def tf_to_matrix(tf_msg: TransformStamped):
    """
    Convert TransformStamped message to a 4x4 transformation matrix.

    Arguments:
        tf_msg (TransformStamped): TransformStamped message.

    Returns:
        numpy.ndarray:
            4x4 transformation matrix representing the translation and rotation.
    """
    # Extract translation values
    xyz = [tf_msg.transform.translation.x,
           tf_msg.transform.translation.y,
           tf_msg.transform.translation.z]

    # Extract rotation values
    q = [tf_msg.transform.rotation.x,
         tf_msg.transform.rotation.y,
         tf_msg.transform.rotation.z,
         tf_msg.transform.rotation.w]

    return xyzq_to_matrix(xyz, q)


def odometry_to_matrix(odom_msg: Odometry):
    """
    Convert Odometry message to a 4x4 transformation matrix.

    Arguments:
        odom_msg (Odometry): Odometry message.

    Returns:
        numpy.ndarray:
            4x4 transformation matrix representing the position and orientation.
    """
    # Extract position
    xyz = [odom_msg.pose.pose.position.x,
           odom_msg.pose.pose.position.y,
           odom_msg.pose.pose.position.z]

    # Extract orientation as a quaternion
    q = [odom_msg.pose.pose.orientation.x,
         odom_msg.pose.pose.orientation.y,
         odom_msg.pose.pose.orientation.z,
         odom_msg.pose.pose.orientation.w]

    return xyzq_to_matrix(xyz, q)


def extract_translation_and_rotation_from_matrix(matrix):
    """
    Extract the translation and rotation from a 4x4 transformation matrix.

    Arguments:
        matrix (numpy.ndarray, (4, 4)):
            Transformation matrix.

    Returns:
        tuple:
            Extracted translation as (x, y, z).
            Extracted rotation as (roll, pitch, yaw).
    """
    translation = matrix[:3, 3]
    rotation = R.from_matrix(matrix[:3, :3]).as_euler('xyz')
    return translation, rotation


def odom_to_matrix(odom_x, odom_y, odom_theta):
    """
    Generate a 2D transformation matrix from pose.

    Arguments:
        odom_x (float): X-coordinate of the pose.
        odom_y (float): Y-coordinate of the pose.
        odom_theta (float): Rotation counterclockwise about the origin.

    Returns:
        numpy.ndarray:
            2D transformation matrix.
    """
    matrix = [
        [np.cos(odom_theta), -np.sin(odom_theta), odom_x],
        [np.sin(odom_theta), np.cos(odom_theta), odom_y],
        [0, 0, 1]
    ]
    return np.array(matrix)


def extract_laser_config(laser_msg: LaserScan):
    """
    Extract LiDAR configuration from LaserScan message.

    Arguments:
        laser_msg (LaserScan): LaserScan message.

    Returns:
        dict:
            Dictionary containing LiDAR configuration information.
    """
    laser_config = {
        'num_beams': len(laser_msg.ranges),
        'angle_min': laser_msg.angle_min,
        'angle_max': laser_msg.angle_max,
        'angle_increment': laser_msg.angle_increment,
        'range_min': laser_msg.range_min,
        'range_max': laser_msg.range_max,
    }

    return laser_config
