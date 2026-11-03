"""TB3 Gazebo 월드 + 상단 카메라 + 대시보드 + ESP32 serial_bridge + teleop_node를 한 번에 실행.

포트 변경: ros2 launch gesture_teleop sim.launch.py port:=/dev/ttyACM1
Gazebo 창 보기: ... gui:=true  (기본은 꺼짐. 영상은 대시보드에서 확인)
turtlebot3_world.launch.py는 gzclient를 항상 띄우므로 그 안의 구성(gzserver, rsp, spawn)을 직접 include한다.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    tb3 = get_package_share_directory('turtlebot3_gazebo')
    gz = get_package_share_directory('gazebo_ros')

    def include(pkg_dir, name, condition=None, **args):
        return IncludeLaunchDescription(PythonLaunchDescriptionSource(os.path.join(pkg_dir, 'launch', name)),
                                        launch_arguments=args.items(), condition=condition)
    me = get_package_share_directory('gesture_teleop')
    return LaunchDescription([
        DeclareLaunchArgument('port', default_value='/dev/ttyACM0'),
        DeclareLaunchArgument('gui', default_value='false'),
        SetEnvironmentVariable('TURTLEBOT3_MODEL', 'burger'),
        include(gz, 'gzserver.launch.py', world=os.path.join(tb3, 'worlds', 'turtlebot3_world.world')),
        include(gz, 'gzclient.launch.py', condition=IfCondition(LaunchConfiguration('gui'))),
        include(tb3, 'robot_state_publisher.launch.py', use_sim_time='true'),
        include(tb3, 'spawn_turtlebot3.launch.py', x_pose='-2.0', y_pose='-0.5'),
        Node(package='gazebo_ros', executable='spawn_entity.py',
             arguments=['-entity', 'overhead_cam', '-file', os.path.join(me, 'overhead_cam.sdf')]),
        Node(package='gesture_teleop', executable='dashboard'),
        Node(package='gesture_teleop', executable='serial_bridge',
             parameters=[{'port': LaunchConfiguration('port')}]),
        Node(package='gesture_teleop', executable='teleop_node'),
    ])
