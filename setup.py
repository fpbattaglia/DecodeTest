"""
Setup script for replay_decoder package.

Based on Liu et al. (2019) Cell: "Human replay spontaneously reorganizes experience"
"""

from setuptools import setup, find_packages
import os

# Read the long description from README
def read_long_description():
    here = os.path.abspath(os.path.dirname(__file__))
    readme_path = os.path.join(here, 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ''

setup(
    name='replay_decoder',
    version='0.1.0',
    author='Based on Liu et al. (2019)',
    author_email='',
    description='Multivariate decoding analysis for neural replay detection',
    long_description=read_long_description(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/replay_decoder',  # Update with actual URL
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    keywords='neuroscience MEG EEG replay decoding hippocampus sequence-learning',
    python_requires='>=3.7',
    install_requires=[
        'numpy>=1.19.0',
        'scipy>=1.5.0',
        'scikit-learn>=0.24.0',
        'matplotlib>=3.3.0',
    ],
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.10',
            'jupyter>=1.0',
            'seaborn>=0.11',
        ],
        'notebook': [
            'jupyter>=1.0',
            'seaborn>=0.11',
        ],
    },
    package_data={
        'replay_decoder': [],
    },
    entry_points={
        'console_scripts': [],
    },
    project_urls={
        'Original Paper': 'https://doi.org/10.1016/j.cell.2019.06.012',
        'Bug Reports': 'https://github.com/yourusername/replay_decoder/issues',
        'Source': 'https://github.com/yourusername/replay_decoder',
    },
)
