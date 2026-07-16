from setuptools import setup

package_name = "smolvla_bridge"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", ["launch/stage1.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Tao Wang",
    maintainer_email="aidenwang752@gmail.com",
    description="SimEnv gRPC to ROS2 bridge owning the 50 Hz tick contract.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "sim_bridge = smolvla_bridge.sim_bridge:main",
            "event_recorder = smolvla_bridge.event_recorder:main",
        ],
    },
)
