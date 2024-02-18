from setuptools import find_packages, setup

setup(
    name="uav_sim",
    author="richzhang@berkeley.edu",
    keywords=["robotics", "rl"],
    packages=find_packages("."),
    install_requires=[
        "numpy",
        "matplotlib",
        'scipy',
        'onnx',
        'onnxruntime',
        'pandas'
    ],
)
