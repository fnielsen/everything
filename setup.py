from setuptools import setup

setup(
    name='everything',
    version='0.1',
    author='Finn Aarup Nielsen',
    author_email='faan@dtu.dk',
    description='Import many modules',
    license='GPL',
    keywords='import',
    url='https://github.com/fnielsen/everything',
    py_modules=['everything'],
    long_description='',
    classifiers=[
        'Topic :: Utilities',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.2',
        ],
    tests_require=['pytest'],
    test_suite='py.test',
    )
