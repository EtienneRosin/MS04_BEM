from setuptools import setup, find_packages

setup(
    name="MS04_PROJECT",
    version="0.1",
    description="A quick BEM solver for the 2D Helmoltz equation in the case of the diffraction of a plane wave by a disc.",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    # url="https://github.com/votreusername/maggiver",
    author="Etienne Rosin",
    author_email="etienne.rosin@ensta-paris.fr",
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'cmasher',
        'scienceplots'
        ],
    include_package_data=True,
)
