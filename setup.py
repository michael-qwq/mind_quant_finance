import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

VERSION = '0.0.1'
PACKAGE_NAME = 'MindQuantFinance'
AUTHOR = 'The MindQuantFinance Authors'
URL = 'https://gitee.com/luweizheng/mind-quant-finance'

LICENSE = 'MIT'
DESCRIPTION = 'AI Accelerated Quantative Finance Library'
LONG_DESCRIPTION = (HERE / "README.md").read_text()
LONG_DESC_TYPE = "text/markdown"

INSTALL_REQUIRES = ['numpy']
TESTS_REQUIRES = ['pytest']

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type=LONG_DESC_TYPE,
    author=AUTHOR,
    license=LICENSE,
    url=URL,
    install_requires=INSTALL_REQUIRES,
    extras_require={
        'tests': TESTS_REQUIRES,
        'complete': INSTALL_REQUIRES + TESTS_REQUIRES,
    },
    packages=find_packages(),
    python_requires='>=3'
)
