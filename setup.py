from setuptools import find_packages, setup

def get_requirements():
    with open("requirements.txt") as f:
        return [
            line.strip() for line in f
            if line.strip() and not line.startswith("#")
        ]

setup(
    name='objectdetection',
    version='0.0.0',
    author='shashank chauhan',
    author_email='shashankchauhan8577@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements()
)