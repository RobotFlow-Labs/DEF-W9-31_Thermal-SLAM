# Thermal-SLAM ROS2 Topic Contract (Draft)

## Subscriptions
- `/thermal/image` (`sensor_msgs/msg/Image`, mono16 expected)

## Publications
- `/thermal/depth` (`sensor_msgs/msg/Image`, 32FC1)
- `/thermal/refined` (`sensor_msgs/msg/Image`, mono8)

## Parameters
- model_path
- seq_len
- image_height
- image_width
- percent_low
- percent_high
