"""
In this script, we are going to install the required packages for the project.

At the same time, we are going to create a virtual environment for the project, that 
will be used to install the packages.
It will be called `venv` and will appear when using a jupyter notebook environment.
"""


def install(packages):
    for package in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])


if __name__ == "__main__":

    import os
    import subprocess
    import sys

    os.system("python3 -m venv venv")
    os.system("source venv/bin/activate")
    print(f"Current Environment is: {sys.executable}")
    os.system("pip install --upgrade pip")
    os.system("./venv/bin/python -m pip install -r requirements.txt")
