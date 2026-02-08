"""
Configuration Management for LCA System

Loads environment variables and provides path resolution utilities.
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load .env file from project root
project_root = Path(__file__).parent.parent.parent
env_file = project_root / ".env"

if env_file.exists():
    load_dotenv(env_file)
else:
    # Try .env.example
    env_example = project_root / ".env.example"
    if env_example.exists():
        load_dotenv(env_example)


class LCAConfig:
    """Configuration manager for LCA system"""

    # Project paths
    PROJECT_ROOT = project_root
    DATA_DIR = PROJECT_ROOT / "data"
    ONTOLOGY_DIR = DATA_DIR / "lca_ontologies"

    # Ontology paths from environment variables
    SNOMED_CT_PATH = os.getenv(
        "SNOMED_CT_PATH",
        str(ONTOLOGY_DIR / "snomed_ct" / "build_snonmed_owl" / "snomed_ct_optimized.owl")
    )

    # SNOMED RF2 Snapshot directory (tab-delimited concept/description/relationship files)
    SNOMED_RF2_PATH = os.getenv(
        "SNOMED_RF2_PATH",
        str(ONTOLOGY_DIR / "snomed_ct" / "SnomedCT_InternationalRF2_PRODUCTION_20260101T120000Z" / "Snapshot" / "Terminology")
    )

    # SNOMED Turtle (converted via ROBOT)
    SNOMED_TTL_PATH = os.getenv(
        "SNOMED_TTL_PATH",
        str(ONTOLOGY_DIR / "snomed_ct" / "robot" / "snomed_ct_optimized.ttl")
    )

    # SHACL shapes for ontology validation
    SHACL_SHAPES_PATH = os.getenv(
        "SHACL_SHAPES_PATH",
        str(ONTOLOGY_DIR / "shacl" / "oncology_shapes.ttl")
    )

    # NCI Thesaurus OWL
    NCIT_OWL_PATH = os.getenv(
        "NCIT_OWL_PATH",
        str(ONTOLOGY_DIR / "ncit" / "ncit.owl")
    )

    LOINC_PATH = os.getenv(
        "LOINC_PATH",
        str(ONTOLOGY_DIR / "loinc" / "Loinc_2.81")
    )

    RXNORM_PATH = os.getenv(
        "RXNORM_PATH",
        str(ONTOLOGY_DIR / "rxnorm" / "RxNorm_full_01052026")
    )

    # UMLS MRCONSO.RRF for crosswalk
    UMLS_PATH = os.getenv(
        "UMLS_PATH",
        str(ONTOLOGY_DIR / "umls")
    )

    LUCADA_ONTOLOGY_OUTPUT = os.getenv(
        "LUCADA_ONTOLOGY_OUTPUT",
        str(ONTOLOGY_DIR / "lucada")
    )

    LUCADA_OWL_FILE = os.getenv(
        "LUCADA_OWL_FILE",
        "lucada_ontology.owl"
    )

    @classmethod
    def get_lucada_output_path(cls) -> Path:
        """Get full path to LUCADA ontology output file"""
        output_dir = Path(cls.LUCADA_ONTOLOGY_OUTPUT)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / cls.LUCADA_OWL_FILE

    # Neo4j Configuration
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
    NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "lucada")
    NEO4J_VECTOR_INDEX = os.getenv("NEO4J_VECTOR_INDEX", "clinical_guidelines_vector")

    # Ollama Configuration
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:latest")

    # MCP Server Configuration
    MCP_SERVER_PORT = int(os.getenv("MCP_SERVER_PORT", "3000"))
    MCP_SERVER_HOST = os.getenv("MCP_SERVER_HOST", "localhost")

    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    @classmethod
    def validate_paths(cls) -> dict:
        """
        Validate that all required ontology files exist.

        Returns:
            Dictionary with validation results
        """
        results = {
            "SNOMED_CT": {
                "path": cls.SNOMED_CT_PATH,
                "exists": Path(cls.SNOMED_CT_PATH).exists()
            },
            "SNOMED_RF2": {
                "path": cls.SNOMED_RF2_PATH,
                "exists": Path(cls.SNOMED_RF2_PATH).exists()
            },
            "SNOMED_TTL": {
                "path": cls.SNOMED_TTL_PATH,
                "exists": Path(cls.SNOMED_TTL_PATH).exists()
            },
            "SHACL_SHAPES": {
                "path": cls.SHACL_SHAPES_PATH,
                "exists": Path(cls.SHACL_SHAPES_PATH).exists()
            },
            "NCIT_OWL": {
                "path": cls.NCIT_OWL_PATH,
                "exists": Path(cls.NCIT_OWL_PATH).exists()
            },
            "LOINC": {
                "path": cls.LOINC_PATH,
                "exists": Path(cls.LOINC_PATH).exists()
            },
            "RXNORM": {
                "path": cls.RXNORM_PATH,
                "exists": Path(cls.RXNORM_PATH).exists()
            },
            "LUCADA_OUTPUT": {
                "path": cls.LUCADA_ONTOLOGY_OUTPUT,
                "exists": Path(cls.LUCADA_ONTOLOGY_OUTPUT).exists()
            },
            "UMLS": {
                "path": cls.UMLS_PATH,
                "exists": Path(cls.UMLS_PATH).exists()
            }
        }
        return results

    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("=" * 80)
        print("LCA SYSTEM CONFIGURATION")
        print("=" * 80)
        print(f"\nProject Root: {cls.PROJECT_ROOT}")
        print(f"Data Directory: {cls.DATA_DIR}")
        print(f"\nOntology Paths:")
        print(f"  SNOMED-CT: {cls.SNOMED_CT_PATH}")
        print(f"  LOINC:     {cls.LOINC_PATH}")
        print(f"  RxNorm:    {cls.RXNORM_PATH}")
        print(f"  LUCADA Output: {cls.get_lucada_output_path()}")
        print(f"\nNeo4j Configuration:")
        print(f"  URI:      {cls.NEO4J_URI}")
        print(f"  Database: {cls.NEO4J_DATABASE}")
        print(f"  User:     {cls.NEO4J_USER}")
        print(f"\nOllama Configuration:")
        print(f"  URL:   {cls.OLLAMA_BASE_URL}")
        print(f"  Model: {cls.OLLAMA_MODEL}")
        print("\nPath Validation:")
        validation = cls.validate_paths()
        for name, info in validation.items():
            status = "✅ EXISTS" if info["exists"] else "❌ MISSING"
            print(f"  {name:20} {status:12} {info['path']}")
        print("=" * 80)


# Create singleton instance
config = LCAConfig()


if __name__ == "__main__":
    # Print configuration when run directly
    LCAConfig.print_config()
