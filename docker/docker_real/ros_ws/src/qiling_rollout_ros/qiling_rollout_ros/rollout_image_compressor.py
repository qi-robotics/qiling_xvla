#!/usr/bin/env python3
"""Robot-side throttled JPEG transport for the three rollout RGB cameras."""

from __future__ import annotations

import time

import cv2
from cv_bridge import CvBridge, CvBridgeError
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image


class RolloutImageCompressor(Node):
    """Compress local raw camera frames before they cross the robot Wi-Fi link."""

    def __init__(self) -> None:
        super().__init__("qiling_rollout_image_compressor")
        self.declare_parameter("output_rate_hz", 15.0)
        self.declare_parameter("jpeg_quality", 80)
        defaults = {
            "head": (
                "/camera_head/head_camera/color/image_raw",
                "/qiling_rollout/camera_head/image/compressed",
            ),
            "left": (
                "/camera_left/left_camera/color/image_raw",
                "/qiling_rollout/camera_left/image/compressed",
            ),
            "right": (
                "/camera_right/right_camera/color/image_raw",
                "/qiling_rollout/camera_right/image/compressed",
            ),
        }
        for name, (input_topic, output_topic) in defaults.items():
            self.declare_parameter(f"{name}_input_topic", input_topic)
            self.declare_parameter(f"{name}_output_topic", output_topic)

        output_rate = float(self.get_parameter("output_rate_hz").value)
        if not 1.0 <= output_rate <= 30.0:
            raise RuntimeError("output_rate_hz must be in [1, 30]")
        self._period = 1.0 / output_rate
        self._jpeg_quality = int(self.get_parameter("jpeg_quality").value)
        if not 1 <= self._jpeg_quality <= 100:
            raise RuntimeError("jpeg_quality must be in [1, 100]")

        self._bridge = CvBridge()
        # RealSense publishes these raw streams as RELIABLE.  A depth-one
        # reliable reader avoids retaining stale full-resolution frames while
        # matching the camera writer exactly.  Compressed frames are also
        # reliable because a dropped DDS fragment invalidates the whole JPEG.
        self._image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        # Serialise callbacks from one camera while allowing the three cameras
        # to encode concurrently in the multi-threaded executor.
        self._image_callback_groups = {
            name: MutuallyExclusiveCallbackGroup() for name in defaults
        }
        self._last_publish = {name: float("-inf") for name in defaults}
        self._received = {name: 0 for name in defaults}
        self._published = {name: 0 for name in defaults}
        self._image_publishers = {}
        self._image_subscriptions = []
        for name in defaults:
            input_topic = str(self.get_parameter(f"{name}_input_topic").value)
            output_topic = str(self.get_parameter(f"{name}_output_topic").value)
            self._image_publishers[name] = self.create_publisher(
                CompressedImage, output_topic, self._image_qos)
            self._image_subscriptions.append(self.create_subscription(
                Image,
                input_topic,
                self._make_callback(name),
                self._image_qos,
                callback_group=self._image_callback_groups[name],
            ))
            self.get_logger().info(f"{name}: {input_topic} -> {output_topic}")

        self.create_timer(5.0, self._diagnostics)
        self.get_logger().info(
            f"Rollout JPEG transport ready: output_rate={output_rate:.1f} Hz, "
            f"jpeg_quality={self._jpeg_quality}")

    def _make_callback(self, name: str):
        def callback(message: Image) -> None:
            self._received[name] += 1
            now = time.monotonic()
            if now - self._last_publish[name] < self._period:
                return
            # Reserve this output slot before encoding so a slow encode cannot
            # cause callback bursts and defeat the configured rate limit.
            self._last_publish[name] = now
            try:
                image_bgr = self._bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
                success, encoded = cv2.imencode(
                    ".jpg",
                    image_bgr,
                    [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality],
                )
                if not success:
                    raise ValueError("cv2.imencode returned false")
            except (CvBridgeError, ValueError, cv2.error) as error:
                self.get_logger().error(f"{name} JPEG encoding failed: {error}")
                return

            output = CompressedImage()
            output.header = message.header
            output.format = "jpeg"
            output.data = encoded.tobytes()
            self._image_publishers[name].publish(output)
            self._published[name] += 1

        return callback

    def _diagnostics(self) -> None:
        summary = ", ".join(
            f"{name}:rx={self._received[name]} tx={self._published[name]}"
            for name in ("head", "left", "right")
        )
        self.get_logger().info(f"JPEG transport counters: {summary}")


def main() -> None:
    rclpy.init()
    node = RolloutImageCompressor()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            executor.shutdown()
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        except (KeyboardInterrupt, ExternalShutdownException):
            pass


if __name__ == "__main__":
    main()
