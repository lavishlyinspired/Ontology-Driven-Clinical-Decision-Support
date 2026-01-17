"""
Setup Script for Lung Cancer Assistant
Automated installation and configuration

Note: This script is now replaced by: python cli.py setup
This file remains for backward compatibility.
"""

import subprocess
import sys

def main():
    print("\n" + "="*80)
    print("LUNG CANCER ASSISTANT - SETUP")
    print("="*80)
    print("\nThis script has been replaced by the new CLI.")
    print("Please run: python cli.py setup")
    print("\nAlternatively, running basic setup now...")
    print("="*80)
    
    # Run the new CLI setup command
    try:
        subprocess.run([sys.executable, "cli.py", "setup"], check=True)
    except subprocess.CalledProcessError:
        print("\n⚠ CLI setup failed. Falling back to manual setup...")
        # Fallback to basic pip install
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("\n✓ Dependencies installed")
        print("\nNext steps:")
        print("  python cli.py check    # Verify services")
        print("  python cli.py info     # Project information")


if __name__ == "__main__":
    main()
