from setuptools import setup, find_packages

setup(name='limbus',
      version='0.1.0',
      description='High level interface to create Pytorch Graphs.',
      author='Kornia.org',
      url='https://github.com/kornia/limbus',
      install_requires=[],
      extras_require={
          'dev': [
              'pytest',
              'pytest-flake8',
              'pytest-cov',
              'pytest-mypy',
              'pytest-pydocstyle',
              'mypy',
              'pydocstyle',
              'flake8',
              'pep8-naming'
          ]
      },
      packages=find_packages(),
      )
