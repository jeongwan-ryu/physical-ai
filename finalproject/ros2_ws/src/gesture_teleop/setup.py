from setuptools import setup

setup(
    name='gesture_teleop',
    version='0.1.0',
    packages=['gesture_teleop'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/gesture_teleop']),
        ('share/gesture_teleop', ['package.xml', 'models/overhead_cam.sdf']),
        ('share/gesture_teleop/launch', ['launch/sim.launch.py']),
    ],
    install_requires=['setuptools'],
    entry_points={'console_scripts': [
        'teleop_node = gesture_teleop.teleop_node:main',
        'serial_bridge = gesture_teleop.serial_bridge:main',
        'dashboard = gesture_teleop.dashboard:main',
    ]},
)
