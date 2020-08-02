try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

    config = {
        'description': 'Convert DDSM normal cases from raw to tiff',
        'author': 'Francisco Gimenez, Eugene Vorontsov',
        'url': 'https://github.com/fjeg/ddsm_tools',
        'author_email': 'fgimenez@stanford.edu, eugene.vorontsov@gmail.com',
        'version': '0.1',
        'packages': ['ddsm_normals'],
        'name': 'ddsm_normals'
    }

setup(**config)
