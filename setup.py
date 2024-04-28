from setuptools import setup, find_packages

setup(
    name="llm.pth",
    version="0.1",
    packages=find_packages(),
    author="abideenml",
    author_email="zaiinn440@gmail.com",
    description="Implementation of various Autoregressive models, Research papers and techniques.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/abideenml/llm.pth",
    install_requires=[
        "transformers",
        "rich",
        "lightning",
        "datasets",
        "torch",
        "wandb",
    ],
    python_requires=">=3.8",
)