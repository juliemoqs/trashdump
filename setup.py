from setuptools import setup, find_packages

setup(
    name="trashdump",
    version="0.1.0",
    author="Robby Wilson",
    author_email="robert.f.wilson@nasa.gov",
    description="TrASHDUMP is a tool for detecting and vettting transiting planets in high-cadence space-based data such as Kepler and TESS",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/robertfwilson/trashdump",  # repository url
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Or whichever license you choose
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",  # or whatever your code supports
    entry_points={
        "console_scripts": [
            "trashdump=trashdump.trashdump_script:main",  # assuming main() exists
        ],
    },
)
