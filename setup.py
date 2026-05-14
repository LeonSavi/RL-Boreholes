from setuptools import setup, find_packages

setup(
    name="decision_simulator",
    version="0.1.0",
    description="Modular chess tournament framework with engines and LLM players",
    packages=find_packages(),
    install_requires=[
        "requests==2.32.4",
    ],
    python_requires=">=3.9",
)
