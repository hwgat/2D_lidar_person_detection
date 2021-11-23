import sys
sys.path.reverse()

# import time
import numpy as np
import rospy

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Point, Pose, PoseArray
from visualization_msgs.msg import Marker
from spencer_tracking_msgs.msg import DetectedPerson, DetectedPersons

from dr_spaam.detector import Detector


class DrSpaamROS:
    """ROS node to detect pedestrian using DROW3 or DR-SPAAM."""

    def __init__(self):
        self._read_params()
        self._detector = Detector(
            self.weight_file,
            model=self.detector_model,
            gpu=False,
            stride=self.stride,
            panoramic_scan=self.panoramic_scan,
        )
        self._init()
        self.detection_id = 0


    def _read_params(self):
        """
        @brief      Reads parameters from ROS server.
        """
        self.weight_file = rospy.get_param("~weight_file")
        self.conf_thresh = rospy.get_param("~conf_thresh")
        self.stride = rospy.get_param("~stride")
        self.detector_model = rospy.get_param("~detector_model")
        self.panoramic_scan = rospy.get_param("~panoramic_scan")

    def _init(self):
        """
        @brief      Initialize ROS connection.
        """
        # Publisher
        topic, queue_size, latch = read_publisher_param("detections")
        self._dets_pub = rospy.Publisher(
            topic, PoseArray, queue_size=queue_size, latch=latch
        )

        topic, queue_size, latch = read_publisher_param("rviz")
        self._rviz_pub = rospy.Publisher(
            topic, Marker, queue_size=queue_size, latch=latch
        )

        topic, queue_size, latch = read_publisher_param("spencer")
        self._spencer_pub = rospy.Publisher(
            topic, DetectedPersons, queue_size=queue_size, latch=latch
        )

        # Subscriber
        topic, queue_size = read_subscriber_param("scan")
        self._scan_sub = rospy.Subscriber(
            topic, LaserScan, self._scan_callback, queue_size=queue_size
        )

    def _scan_callback(self, msg):
        if (
            self._dets_pub.get_num_connections() == 0
            and self._rviz_pub.get_num_connections() == 0
            and self._spencer_pub.get_num_connections() == 0
        ):
            return

        # TODO check the computation here
        if not self._detector.is_ready():
            self._detector.set_laser_fov(
                np.rad2deg(msg.angle_increment * len(msg.ranges))
            )

        scan = np.array(msg.ranges)
        scan[scan == 0.0] = 29.99
        scan[np.isinf(scan)] = 29.99
        scan[np.isnan(scan)] = 29.99

        # t = time.time()
        dets_xy, dets_cls, _ = self._detector(scan)
        # print("[DrSpaamROS] End-to-end inference time: %f" % (t - time.time()))

        # confidence threshold
        conf_mask = (dets_cls >= self.conf_thresh).reshape(-1)
        dets_xy = dets_xy[conf_mask]
        dets_cls = dets_cls[conf_mask]
        # convert to ros msg and publish
        dets_msg = detections_to_pose_array(dets_xy, dets_cls)
        dets_msg.header = msg.header
        self._dets_pub.publish(dets_msg)

        rviz_msg = detections_to_rviz_marker(dets_xy, dets_cls)
        rviz_msg.header = msg.header
        self._rviz_pub.publish(rviz_msg)

        spencer_msg = detections_to_spencer_detections(dets_xy, dets_cls,self.detection_id)
        spencer_msg.header = msg.header
        spencer_msg.header.stamp = rospy.Time.now()
        self.detection_id += len(spencer_msg.detections)
        self._spencer_pub.publish(spencer_msg)


def detections_to_rviz_marker(dets_xy, dets_cls):
    """
    @brief     Convert detection to RViz marker msg. Each detection is marked as
               a circle approximated by line segments.
    """
    msg = Marker()
    msg.action = Marker.ADD
    msg.ns = "dr_spaam_ros"
    msg.id = 0
    msg.type = Marker.LINE_LIST

    # set quaternion so that RViz does not give warning
    msg.pose.orientation.x = 0.0
    msg.pose.orientation.y = 0.0
    msg.pose.orientation.z = 0.0
    msg.pose.orientation.w = 1.0

    msg.scale.x = 0.03  # line width
    # red color
    msg.color.r = 1.0
    msg.color.a = 1.0

    # circle
    r = 0.4
    ang = np.linspace(0, 2 * np.pi, 20)
    xy_offsets = r * np.stack((np.cos(ang), np.sin(ang)), axis=1)

    # to msg
    for d_xy, d_cls in zip(dets_xy, dets_cls):
        for i in range(len(xy_offsets) - 1):
            # start point of a segment
            p0 = Point()
            p0.x = d_xy[0] + xy_offsets[i, 0]
            p0.y = d_xy[1] + xy_offsets[i, 1]
            p0.z = 0.0
            msg.points.append(p0)

            # end point
            p1 = Point()
            p1.x = d_xy[0] + xy_offsets[i + 1, 0]
            p1.y = d_xy[1] + xy_offsets[i + 1, 1]
            p1.z = 0.0
            msg.points.append(p1)

    return msg


def detections_to_pose_array(dets_xy, dets_cls):
    pose_array = PoseArray()
    for d_xy, d_cls in zip(dets_xy, dets_cls):
        # Detector uses following frame convention:
        # x forward, y rightward, z downward, phi is angle w.r.t. x-axis
        p = Pose()
        p.position.x = d_xy[0]
        p.position.y = d_xy[1]
        p.position.z = 0.0
        pose_array.poses.append(p)

    return pose_array

def detections_to_spencer_detections(dets_xy, dets_cls,detection_id):
    ped_locations = DetectedPersons()
    id_num = detection_id
    for d_xy, d_cls in zip(dets_xy, dets_cls):
        ped_location = DetectedPerson()
        id_num+=1
        ped_location.detection_id = id_num
        ped_location.confidence = d_cls
        ped_location.pose.pose.position.x = d_xy[0]
        ped_location.pose.pose.position.y = d_xy[1]
        ped_location.pose.pose.position.z = 0.0
        ped_location.pose.covariance = [0.11349243571626173, -0.013492485277898883, 3673.2050992363506, 0.0, 0.0, 0.0, -0.013492485277898881, 0.11349253483971816, -3673.218591986086, 0.0, 0.0, 0.0, 3673.2050992363506, -3673.218591986086, 999999998.9730148, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 999999.0, 999999.0, 999999.0, 0.0, 0.0, 0.0, 999999.0, 999999.0, 999999.0, 0.0, 0.0, 0.0, 999999.0, 999999.0, 999999.0]
        ped_location.modality = 'laser2d'
        ped_locations.detections.append(ped_location)

    return ped_locations


def read_subscriber_param(name):
    """
    @brief      Convenience function to read subscriber parameter.
    """
    topic = rospy.get_param("~subscriber/" + name + "/topic")
    queue_size = rospy.get_param("~subscriber/" + name + "/queue_size")
    return topic, queue_size


def read_publisher_param(name):
    """
    @brief      Convenience function to read publisher parameter.
    """
    topic = rospy.get_param("~publisher/" + name + "/topic")
    queue_size = rospy.get_param("~publisher/" + name + "/queue_size")
    latch = rospy.get_param("~publisher/" + name + "/latch")
    return topic, queue_size, latch
