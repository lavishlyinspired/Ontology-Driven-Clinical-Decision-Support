#!/usr/bin/env python3
"""
Startup script for LCA MCP Server
Validates environment and starts the server
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def check_prerequisites():
    """Check if all prerequisites are met"""
    print("=" * 60)
    print("LCA MCP Server - Prerequisites Check")
    print("=" * 60)

    issues = []

    # Check Python version
    if sys.version_info < (3, 9):
        issues.append(f"Python 3.9+ required, found {sys.version}")
    else:
        print(f"✓ Python version: {sys.version.split()[0]}")

    # Check required packages
    required_packages = [
        "mcp",
        "pydantic",
        "owlready2",
        "langchain",
        "neo4j"
    ]

    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ Package {package} installed")
        except ImportError:
            issues.append(f"Package {package} not installed")
            print(f"✗ Package {package} NOT installed")

    # Check .env file
    env_file = project_root / ".env"
    if env_file.exists():
        print(f"✓ .env file found at {env_file}")
    else:
        issues.append(".env file not found")
        print(f"✗ .env file NOT found at {env_file}")

    # Check data directory
    data_dir = project_root / "data"
    if data_dir.exists():
        print(f"✓ Data directory exists")
    else:
        print(f"⚠ Data directory not found at {data_dir} (optional ontologies)")

    print("=" * 60)

    if issues:
        print("\n⚠ Issues found:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nThe server may still start but some features will be unavailable.")
        print("=" * 60)
    else:
        print("✓ All prerequisites met!")
        print("=" * 60)

    return len(issues) == 0

def main():
    """Main entry point"""
    # Check prerequisites
    check_prerequisites()

    # Import and run server
    print("\nStarting MCP server...\n")

    try:
        from backend.src.mcp_server.lca_mcp_server_adaptive import main as server_main
        import asyncio
        asyncio.run(server_main())
    except KeyboardInterrupt:
        print("\n\nServer stopped by user")
    except Exception as e:
        print(f"\n✗ Error starting server: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
