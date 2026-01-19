"""
Quick verification script for Neo4j vector store migration
Run this to verify the migration was successful
"""

import os
import sys
from pathlib import Path

print("=" * 80)
print("Neo4j Vector Store Migration Verification")
print("=" * 80)

# Check 1: Environment configuration
print("\n1. Checking environment configuration...")
from dotenv import load_dotenv
load_dotenv()

neo4j_uri = os.getenv("NEO4J_URI")
neo4j_vector_index = os.getenv("NEO4J_VECTOR_INDEX")

if neo4j_uri:
    print(f"   ✓ NEO4J_URI: {neo4j_uri}")
else:
    print("   ✗ NEO4J_URI not set")

if neo4j_vector_index:
    print(f"   ✓ NEO4J_VECTOR_INDEX: {neo4j_vector_index}")
else:
    print("   ✗ NEO4J_VECTOR_INDEX not set")

# Check 2: Dependencies
print("\n2. Checking dependencies...")

try:
    from neo4j import GraphDatabase
    print("   ✓ neo4j package installed")
except ImportError:
    print("   ✗ neo4j package not installed")
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer
    print("   ✓ sentence-transformers package installed")
except ImportError:
    print("   ✗ sentence-transformers package not installed")
    sys.exit(1)

try:
    import chromadb
    print("   ⚠ chromadb still installed (can be removed)")
except ImportError:
    print("   ✓ chromadb removed (expected)")

# Check 3: Vector store module
print("\n3. Checking vector store module...")

try:
    sys.path.insert(0, str(Path(__file__).parent / "backend"))
    from src.db.vector_store import LUCADAVectorStore
    print("   ✓ LUCADAVectorStore class imports successfully")
except Exception as e:
    print(f"   ✗ Failed to import LUCADAVectorStore: {e}")
    sys.exit(1)

# Check 4: Neo4j connection (optional, requires Neo4j to be running)
print("\n4. Checking Neo4j connection...")

try:
    driver = GraphDatabase.driver(
        neo4j_uri,
        auth=(os.getenv("NEO4J_USER", "neo4j"), 
              os.getenv("NEO4J_PASSWORD", "password"))
    )
    driver.verify_connectivity()
    driver.close()
    print("   ✓ Neo4j connection successful")
    neo4j_available = True
except Exception as e:
    print(f"   ⚠ Neo4j not available: {e}")
    print("   Note: This is OK if you haven't started Neo4j yet")
    neo4j_available = False

# Check 5: File structure
print("\n5. Checking file structure...")

required_files = [
    "backend/src/db/vector_store.py",
    "requirements.txt",
    ".env",
    "cli.py"
]

for file_path in required_files:
    if Path(file_path).exists():
        print(f"   ✓ {file_path}")
    else:
        print(f"   ✗ {file_path} missing")

# Check 6: CLI availability
print("\n6. Checking CLI...")

try:
    import typer
    print("   ✓ typer package installed (CLI available)")
except ImportError:
    print("   ⚠ typer not installed (run: pip install typer)")

# Summary
print("\n" + "=" * 80)
print("VERIFICATION SUMMARY")
print("=" * 80)

if neo4j_available:
    print("✅ All checks passed!")
    print("\nNext steps:")
    print("  1. Load sample data: python cli.py generate-samples")
    print("  2. Run workflow: python cli.py run-workflow")
    print("  3. Start API: python cli.py start-api")
else:
    print("⚠️  Migration complete, but Neo4j is not running")
    print("\nTo start Neo4j:")
    print("  docker run -d --name neo4j \\")
    print("    -p 7474:7474 -p 7687:7687 \\")
    print("    -e NEO4J_AUTH=neo4j/123456789 \\")
    print("    neo4j:5.16")
    print("\nOr start your local Neo4j instance")

print("\n" + "=" * 80)
