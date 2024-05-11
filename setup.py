from setuptools import find_packages, setup

description = (
	"A python package for the aggregation of collectively,"
	" annotated data using the size-matters principle"
)

setup(
	name="crowd_label_py",
	version="0.1.0",
	description=description,
	url="https://github.com/taharallouche/crowd-label-py",
	author="Tahar Allouche",
	author_email="tahar.allouche.pro@gmail.com",
	license="BSD 2-clause",
	packages=find_packages(include=["crowd_label", "crowd_label.*"]),
	install_requires=[
		"numpy>=1.24.1",
		"scipy>=1.10.0",
		"pandas>=1.4.1",
		"matplotlib>=3.6.3",
		"scikit-learn>=1.2.1",
	],
	test_suite="tests",
	classifiers=[
		"Development Status :: 1 - Planning",
		"Intended Audience :: Science/Research",
		"Programming Language :: Python :: 3.11",
	],
)
