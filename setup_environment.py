"""
Environment Setup Script for LCA System

This script helps validate and configure the environment for the LCA system.
"""

import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend" / "src"))

from config import LCAConfig


def main():
    """Main setup function"""
    print("\n")
    print("=" * 80)
    print("LUNG CANCER ASSISTANT - ENVIRONMENT SETUP")
    print("=" * 80)
    print()

    # Print current configuration
    LCAConfig.print_config()

    # Check for .env file
    env_file = LCAConfig.PROJECT_ROOT / ".env"
    env_example = LCAConfig.PROJECT_ROOT / ".env.example"

    print("\n" + "=" * 80)
    print("ENVIRONMENT FILE CHECK")
    print("=" * 80)

    if env_file.exists():
        print(f"✅ .env file found at: {env_file}")
    else:
        print(f"❌ .env file not found at: {env_file}")
        if env_example.exists():
            print(f"\n💡 Suggestion: Copy .env.example to .env")
            print(f"   Command: copy \"{env_example}\" \"{env_file}\"")
        else:
            print(f"❌ .env.example also not found at: {env_example}")

    # Validate all paths
    print("\n" + "=" * 80)
    print("PATH VALIDATION SUMMARY")
    print("=" * 80)

    validation = LCAConfig.validate_paths()
    all_valid = all(info["exists"] for info in validation.values())

    missing_paths = []
    for name, info in validation.items():
        if not info["exists"]:
            missing_paths.append((name, info["path"]))

    if all_valid:
        print("\n✅ All ontology paths are valid!")
    else:
        print(f"\n❌ {len(missing_paths)} path(s) missing:")
        for name, path in missing_paths:
            print(f"   - {name}: {path}")

    # Provide setup instructions
    print("\n" + "=" * 80)
    print("SETUP INSTRUCTIONS")
    print("=" * 80)

    if not all_valid:
        print("\n📝 To fix missing paths:")
        print("\n1. Ensure you have the ontology files in the correct locations:")
        print(f"   - SNOMED-CT OWL file: {LCAConfig.SNOMED_CT_PATH}")
        print(f"   - LOINC directory:    {LCAConfig.LOINC_PATH}")
        print(f"   - RxNorm directory:   {LCAConfig.RXNORM_PATH}")
        print(f"\n2. Or update your .env file with the correct paths")
        print(f"\n3. Run this script again to validate")

    # Check Python dependencies
    print("\n" + "=" * 80)
    print("PYTHON DEPENDENCIES CHECK")
    print("=" * 80)

    required_packages = [
        ("owlready2", "Ontology manipulation"),
        ("neo4j", "Neo4j database connectivity"),
        ("fastapi", "REST API framework"),
        ("langchain", "LangChain framework"),
        ("langgraph", "LangGraph workflow"),
        ("pydantic", "Data validation"),
        ("python-dotenv", "Environment variable loading")
    ]

    missing_packages = []
    for package, description in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package:20} ({description})")
        except ImportError:
            print(f"❌ {package:20} ({description}) - NOT INSTALLED")
            missing_packages.append(package)

    if missing_packages:
        print(f"\n💡 Install missing packages:")
        print(f"   pip install {' '.join(missing_packages)}")
    else:
        print(f"\n✅ All required Python packages are installed!")

    # Final status
    print("\n" + "=" * 80)
    print("SETUP STATUS")
    print("=" * 80)

    if all_valid and not missing_packages:
        print("\n🎉 SETUP COMPLETE - All checks passed!")
        print("\nYou can now:")
        print("  1. Generate LUCADA ontology:")
        print("     python -m backend.src.ontology.lucada_ontology")
        print("\n  2. Run tests:")
        print("     python run_tests.py")
        print("\n  3. Start the API server:")
        print("     python -m backend.src.api.main")
    else:
        issues = []
        if not all_valid:
            issues.append(f"{len(missing_paths)} missing path(s)")
        if missing_packages:
            issues.append(f"{len(missing_packages)} missing package(s)")

        print(f"\n⚠️  SETUP INCOMPLETE - {', '.join(issues)}")
        print("\nPlease resolve the issues above and run this script again.")

    print("\n" + "=" * 80 + "\n")

    return 0 if (all_valid and not missing_packages) else 1


if __name__ == "__main__":
    sys.exit(main())
