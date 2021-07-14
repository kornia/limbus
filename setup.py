from setuptools import setup, find_packages

setup(name='limbus',
      version='0.1.0',
      description='Directed Acyclic processing Graph (DAG) framework for PyTorch.',
      author='Kornia.org',
      url='https://github.com/kornia/limbus',
      install_requires=['torch'],
      extras_require={
        'dev': [
            'pytest',
            'pytest-pep8',
            'pytest-cov'
        ]
      },
      packages=find_packages(),
     )   