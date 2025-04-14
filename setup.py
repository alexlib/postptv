# -*- coding: utf-8 -*-
"""
Installation script for the Flowtracks package.

@author: yosef
"""

import re
from pathlib import Path
from setuptools import setup, find_packages
from glob import glob

# Read the version from __init__.py
INIT_FILE = Path('flowtracks/__init__.py')
init_content = INIT_FILE.read_text()
version_match = re.search(r"__version__\s*=\s*['\"]([^'\"]*)['\"]\s*", init_content)
if not version_match:
    raise RuntimeError('Cannot find version in flowtracks/__init__.py')
version = version_match.group(1)

setup(
    name='flowtracks',
    version=version,
    description='Library for handling of PTV trajectory database.',
    long_description=Path('README.md').read_text() if Path('README.md').exists() else '',
    long_description_content_type='text/markdown',
    author='Yosef Meller',
    author_email='yosefm@gmail.com',
    url='https://github.com/OpenPTV/postptv',
    packages=find_packages(),
    data_files=[('flowtracks-examples', glob('examples/*'))],
    scripts=['scripts/analyse_fhdf.py'],
    install_requires=[
        'numpy',
        'scipy',
        'tables',
    ],
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering',
    ],
)
