from setuptools import setup, find_packages


def get_version() -> str:
    return "1.0.0"


setup(
    name="safe_rl_with_rare",
    version=get_version(),
    packages=find_packages(),
)
