'''This sets up the package.
'''
from setuptools import setup, find_packages

def readme():
    """Load the README file."""
    with open('README.md') as f:
        return f.read()


with open('requirements.txt') as infd:
    INSTALL_REQUIRES = [x.strip('\n') for x in infd.readlines()]
    print(INSTALL_REQUIRES)

setup(name='kuaizi', 
	description='Picking low surface brightness galaxies in HSC data with kuaizi (chopsticks in Chinese)',
	long_description=readme(),
	long_description_content_type="text/markdown",
	classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
	url='https://github.com/AstroJacobLi/kuaizi',
	author='Jiaxuan Li',
	author_email='jiaxuan_li@pku.edu.cn',
	license='MIT',
	packages=find_packages(),
    package_data={'kuaizi': ['mplstyle/*']},
	install_requires=INSTALL_REQUIRES,
    include_package_data=True,
    zip_safe=False,
    python_requires='>=3.5',
    use_scm_version=True,
    setup_requires=['setuptools_scm'])
