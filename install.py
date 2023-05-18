_all_ = [
    "torch", 
    "torchvision",
    "torchaudio", 
    "torchopt", 
    "numpy",
    "matplotlib",
    "seaborn",    
    "ipykernel",
]

# Create the list of packages to install for each platform individually

linux = []
# The linux installation of torch, torchvision, and torchaudio has cuda support by default

darwin = []

def install(packages):
    for package in packages:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

if __name__ == '__main__':

    import sys
    import subprocess

    install(_all_) 
    if sys.platform.startswith('linux'):
        install(linux)
    if sys.platform == 'darwin': # MacOS
        install(darwin)