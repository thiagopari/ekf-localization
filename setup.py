from setuptools import setup

package_name = 'ekf_localization'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
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
