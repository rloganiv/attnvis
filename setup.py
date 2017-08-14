from setuptools import setup

setup(
    name='attnvis',
    packages=['attnvis'],
    includes_package_data=True,
    install_requires=[
        'flask',
        'mumie'
    ],
)

