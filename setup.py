import os
from setuptools import setup, find_packages


def parse_requirements(file):
    return sorted(set(
        line.partition('#')[0].strip()
        for line in open(os.path.join(os.path.dirname(__file__), file))
    ) - set(''))


setup(
    name='aitlas',
    python_requires='>=3.6',
    version='0.0.1',
    description='AiTLAS toolbox for working with EO data.',
    author='Bias Variance Labs',
    author_email='someone@bvl.com',
    packages=find_packages(),
    install_requires=parse_requirements('requirements.txt'),
    extras_require={
        'DEV': parse_requirements('requirements-dev.txt')
    }
)
