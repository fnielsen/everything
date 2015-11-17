"""Setup everything package."""


from subprocess import PIPE, Popen

from setuptools import setup


try:
    version, err = Popen(['git', 'describe', '--always'],
                         stdout=PIPE).communicate()
    version = str(version).strip()
except:
    version = ''


setup(
    name='everything',
    version='0.1.dev0' + version,
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
