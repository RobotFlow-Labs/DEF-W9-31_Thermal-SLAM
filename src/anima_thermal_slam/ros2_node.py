from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ROS2 node skeleton for Thermal-SLAM module")
    p.add_argument("--thermal_topic", type=str, default="/thermal/image")
    p.add_argument("--depth_topic", type=str, default="/thermal/depth")
    p.add_argument("--refined_topic", type=str, default="/thermal/refined")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import rclpy  # type: ignore  # pragma: no cover
    except Exception:
        print(
            "ROS2 runtime is not installed in this environment. "
            f"Node skeleton validated. thermal_topic={args.thermal_topic}"
        )
        return

    # Real ROS2 integration intentionally deferred; this skeleton keeps interface stable.
    rclpy.init(args=None)
    print("ROS2 is available. Integrate actual node logic in next pass.")
    rclpy.shutdown()


if __name__ == "__main__":
    main()
