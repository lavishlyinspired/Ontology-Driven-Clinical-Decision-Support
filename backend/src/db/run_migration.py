"""
Neo4j Migration Runner
Runs Cypher migration scripts to update the Neo4j schema

Usage:
    python run_migration.py [migration_file]

Example:
    python run_migration.py migrations/001_add_lab_medication_schema.cypher
"""

import os
import sys
from pathlib import Path
from typing import Optional
from neo4j import GraphDatabase
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Neo4jMigrationRunner:
    """Runs Neo4j migration scripts"""

    def __init__(self, uri: str, username: str, password: str):
        """
        Initialize migration runner

        Args:
            uri: Neo4j connection URI (e.g., bolt://localhost:7687)
            username: Neo4j username
            password: Neo4j password
        """
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        logger.info(f"Connected to Neo4j at {uri}")

    def close(self):
        """Close the driver connection"""
        self.driver.close()
        logger.info("Closed Neo4j connection")

    def run_migration(self, migration_file: Path) -> bool:
        """
        Run a Cypher migration script

        Args:
            migration_file: Path to the .cypher migration file

        Returns:
            True if migration successful, False otherwise
        """
        if not migration_file.exists():
            logger.error(f"Migration file not found: {migration_file}")
            return False

        logger.info(f"Running migration: {migration_file.name}")

        # Read the migration script
        with open(migration_file, 'r', encoding='utf-8') as f:
            migration_script = f.read()

        # Split into statements (separated by semicolons)
        statements = self._split_cypher_statements(migration_script)
        logger.info(f"Found {len(statements)} statements to execute")

        # Execute each statement
        with self.driver.session() as session:
            for i, statement in enumerate(statements, 1):
                if not statement.strip():
                    continue

                # Skip comment-only lines
                if statement.strip().startswith('//'):
                    continue

                try:
                    logger.info(f"Executing statement {i}/{len(statements)}")
                    result = session.run(statement)

                    # Log summary if available
                    summary = result.consume()
                    if summary.counters:
                        logger.info(f"  Created: {summary.counters.nodes_created} nodes, "
                                  f"{summary.counters.relationships_created} relationships, "
                                  f"{summary.counters.indexes_added} indexes, "
                                  f"{summary.counters.constraints_added} constraints")

                except Exception as e:
                    logger.error(f"Error executing statement {i}: {e}")
                    logger.error(f"Statement: {statement[:100]}...")
                    return False

        logger.info(f"✅ Migration {migration_file.name} completed successfully!")
        return True

    def _split_cypher_statements(self, script: str) -> list[str]:
        """
        Split Cypher script into individual statements

        Args:
            script: The full Cypher script

        Returns:
            List of individual statements
        """
        # Simple split by semicolon (not perfect but works for most cases)
        # Note: This doesn't handle semicolons in strings properly
        statements = []
        current_statement = []
        in_block_comment = False

        for line in script.split('\n'):
            line = line.strip()

            # Handle block comments
            if line.startswith('/*'):
                in_block_comment = True
            if '*/' in line:
                in_block_comment = False
                continue
            if in_block_comment:
                continue

            # Skip single-line comments
            if line.startswith('//'):
                continue

            # Skip empty lines
            if not line:
                continue

            # Add line to current statement
            current_statement.append(line)

            # If line ends with semicolon, it's the end of a statement
            if line.endswith(';'):
                # Join the statement and remove the trailing semicolon
                statement = ' '.join(current_statement)[:-1]
                statements.append(statement)
                current_statement = []

        # Add any remaining statement
        if current_statement:
            statement = ' '.join(current_statement)
            if statement.strip():
                statements.append(statement)

        return statements

    def verify_migration(self) -> dict:
        """
        Verify the migration by checking indexes and constraints

        Returns:
            Dictionary with verification results
        """
        logger.info("Verifying migration...")

        with self.driver.session() as session:
            # Check indexes
            indexes_result = session.run("SHOW INDEXES")
            indexes = [dict(record) for record in indexes_result]

            # Check constraints
            constraints_result = session.run("SHOW CONSTRAINTS")
            constraints = [dict(record) for record in constraints_result]

            # Count node labels
            labels_result = session.run("""
                CALL db.labels() YIELD label
                CALL apoc.cypher.run('MATCH (n:' + label + ') RETURN count(n) as count', {})
                YIELD value
                RETURN label, value.count as count
            """)
            node_counts = {record['label']: record['count'] for record in labels_result}

        verification = {
            'indexes': len(indexes),
            'constraints': len(constraints),
            'node_counts': node_counts
        }

        logger.info(f"Verification results:")
        logger.info(f"  Indexes: {verification['indexes']}")
        logger.info(f"  Constraints: {verification['constraints']}")
        logger.info(f"  Node counts: {verification['node_counts']}")

        return verification


def main():
    """Main function to run migrations"""
    # Load environment variables
    load_dotenv()

    # Get Neo4j connection details from environment
    neo4j_uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    neo4j_username = os.getenv('NEO4J_USERNAME', 'neo4j')
    neo4j_password = os.getenv('NEO4J_PASSWORD', 'password')

    # Get migration file from command line or use default
    if len(sys.argv) > 1:
        migration_file = Path(sys.argv[1])
    else:
        # Use the latest migration by default
        migrations_dir = Path(__file__).parent / 'migrations'
        migration_files = sorted(migrations_dir.glob('*.cypher'))
        if not migration_files:
            logger.error("No migration files found in migrations directory")
            sys.exit(1)
        migration_file = migration_files[-1]  # Get the latest

    # Run migration
    runner = Neo4jMigrationRunner(neo4j_uri, neo4j_username, neo4j_password)

    try:
        success = runner.run_migration(migration_file)

        if success:
            # Verify migration
            runner.verify_migration()
            logger.info("✅ Migration completed and verified successfully!")
            sys.exit(0)
        else:
            logger.error("❌ Migration failed!")
            sys.exit(1)

    except Exception as e:
        logger.error(f"❌ Migration error: {e}")
        sys.exit(1)

    finally:
        runner.close()


if __name__ == "__main__":
    main()
