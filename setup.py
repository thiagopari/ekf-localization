from setuptools import setup
from glob import glob

package_name = 'ekf_localization'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Thiago',
    maintainer_email='you@email.com',
    description='EKF sensor fusion for robot localization',
    license='MIT',
    entry_points={
        'console_scripts': [
            'ekf_node = ekf_localization.ekf_node:main',
        ],
    },
)
