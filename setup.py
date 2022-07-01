from setuptools import setup, find_packages

setup(name='limbus',
      version='0.1.0',
      description='High level interface to create Pytorch Graphs.',
      author='Kornia.org',
      url='https://github.com/kornia/limbus',
      install_requires=[
          'torch',
          'numpy',
          'visdom',
          'typeguard',
          'kornia',
          'pyyaml',
          'mypy_extensions'  # only required for some python versions (<3.8)because TypedDict is not supported
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
              'pep8-naming',
              'types-PyYAML'
          ]
      },
      packages=find_packages(where='.'),
      package_dir={'': '.'},
      package_data={'': ['*.yml']},
      include_package_data=True
      )
