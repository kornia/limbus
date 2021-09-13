from setuptools import setup, find_packages

setup(name='limbus',
      version='0.1.0',
      description='High level interface to create Pytorch Graphs.',
      author='Kornia.org',
      url='https://github.com/kornia/limbus',
      install_requires=[
          'torch',
          'numpy',
          'matplotlib',
          'tensorboard',
          'visdom',
          'typeguard',
          'kornia'
      ],
      extras_require={
          'dev': [
              'pytest',
              'pytest-flake8',
              'pytest-cov',
              'pytest-mypy',
              'pytest-pydocstyle',
              'mypy',  # TODO: check if we can remove the deps without pytest-*
              'pydocstyle',
              'flake8',
              'pep8-naming'
          ]
      },
      packages=find_packages(),
      )
