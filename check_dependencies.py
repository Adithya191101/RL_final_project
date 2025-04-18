"""
Script to check if all dependencies are installed correctly.
"""
import sys
import importlib.util

def check_package(package_name):
    """Check if a package is installed."""
    spec = importlib.util.find_spec(package_name)
    if spec is None:
        print(f"❌ {package_name} is NOT installed")
        return False
    else:
        try:
            module = importlib.import_module(package_name)
            version = getattr(module, "__version__", "unknown")
            print(f"✅ {package_name} is installed (version: {version})")
            return True
        except ImportError:
            print(f"❌ {package_name} is installed but cannot be imported")
            return False

def check_matplotlib_3d():
    """Check if matplotlib 3D plotting works."""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        # Create a simple 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter([0], [0], [0], c='r', marker='o')
        
        print("✅ Matplotlib 3D plotting works!")
        plt.close(fig)
        return True
    except Exception as e:
        print(f"❌ Matplotlib 3D plotting doesn't work: {str(e)}")
        return False

def main():
    """Main function to check dependencies."""
    print("Checking dependencies...\n")
    
    # List of packages to check
    packages = [
        "numpy",
        "matplotlib",
        "pybullet",
        "gym",
        "torch",
        "yaml",
        "tqdm",
        "scipy"
    ]
    
    # Check each package
    all_installed = True
    for package in packages:
        all_installed &= check_package(package)
    
    print("\nChecking Matplotlib 3D plotting...")
    matplotlib_3d_works = check_matplotlib_3d()
    
    # Print summary
    print("\nSummary:")
    if all_installed:
        print("All packages are installed!")
    else:
        print("Some packages are missing. Please install them using:")
        print("pip install -r requirements.txt")
    
    if not matplotlib_3d_works:
        print("\nMatplotlib 3D plotting doesn't work. Try reinstalling matplotlib:")
        print("pip uninstall matplotlib")
        print("pip install matplotlib")
    
    print("\nPython version:", sys.version)

if __name__ == "__main__":
    main()