from setuptools import setup, find_packages

setup(
    name='bank_marketing_analysis',
    version='0.1',
    packages=find_packages(),
    author='Mahmoud Abdelhafez',
    author_email='mahmoud.abdelhafez212@gmail.com',
    description='Bank marketing campaign analysis project',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/MahmoudAbdelhafezGH/Bank-Market-Segmentation',
    install_requires=[
        'pandas',
        'numpy',
        'torch',
        'scikit-learn',
        'matplotlib'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
