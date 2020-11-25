from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['Keras==2.4.3',
                     'tensorflow==2.2',
                     'scipy==1.4.1',
                     'grpcio>=1.10.3',
                     'six>=1.13.0']

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Keras trainer application'
)