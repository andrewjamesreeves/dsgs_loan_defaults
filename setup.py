#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Note: To use the 'upload' functionality of this file, you must:
#   $ pipenv install twine --dev
import os
import sys
import glob
import re
from shutil import rmtree

from setuptools import find_packages, setup, Command

# Package meta-data.
NAME = 'dsgs_loan_defaults'
DESCRIPTION = 'Graduate scheme loan defaults'
REQUIRES_PYTHON = '>=3.7.0'

#Read in version from _version.py
VERSION_FILE = "pipeline/_version.py"
verstrline = open(VERSION_FILE, "rt").read()
print(verstrline)

# Automatically detect the package version from VERSION_FILE
VERSION_REGEX = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VERSION_REGEX, verstrline, re.M)
if mo:
    version_string = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSION_FILE,))
VERSION = version_string

# What packages are required for this module to be executed?
REQUIRED = [
    "matplotlib",
    "numpy",
    "pandas",
    "scikit-learn",
    "esig",
    "pyreadstat",
]


here = os.path.abspath(os.path.dirname(__file__))

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
    with open(os.path.join(here, project_slug, '__version__.py')) as f:
        exec(f.read(), about)
else:
    about['__version__'] = VERSION


class UploadCommand(Command):
    """Support setup.py upload."""

    description = 'Build and publish the package.'
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status('Removing previous builds…')
            rmtree(os.path.join(here, 'dist'))
        except OSError:
            pass

        self.status('Building Source and Wheel (universal) distribution…')
        os.system('{0} setup.py sdist bdist_wheel --universal'.format(sys.executable))

        self.status('Uploading the package to PyPI via Twine…')
        os.system('twine upload dist/*')

        self.status('Pushing git tags…')
        os.system('git tag v{0}'.format(about['__version__']))
        os.system('git push --tags')

        sys.exit()


# Where the magic happens:
setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description_content_type='text/markdown',
    python_requires=REQUIRES_PYTHON,
    package_dir={"": "pipeline"},
    packages=find_packages(where="pipeline", exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    py_modules=[os.path.splitext(os.path.basename(path))[0] for path in glob.glob('pipeline/*.py')],
   
    install_requires=REQUIRED,
    include_package_data=True,
    # $ setup.py publish support.
    cmdclass={
        'upload': UploadCommand,
    },
)
