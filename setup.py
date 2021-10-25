import setuptools

setuptools.setup(
    name="vision_utils",
    version="0.0.1",
    author="Anh Le-Tuan, Trung-Kien Tran, Duc Manh Nguyen, Jicheng Yuan, Manfred Hauswirth and Danh Le Phuoc",
    author_email="",
    description="Vision Utils",
    long_description="",
    long_description_content_type="text/markdown",
    url="https://github.com/cqels/vision.git",
    project_urls={
        "Bug Tracker": "https://github.com/cqels/vision/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
	install_requires=[
            'funcy',
            'coloredlogs'
            ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)
