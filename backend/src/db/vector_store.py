"""
Vector Store Integration for LUCADA
Provides semantic search over clinical guidelines and literature
"""

from neo4j import GraphDatabase
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LUCADAVectorStore:
    """
    Vector store for semantic search over clinical guidelines.
    Uses Neo4j vector indexes with sentence-transformers embeddings.
    """

    def __init__(
        self,
        neo4j_uri: str = None,
        neo4j_user: str = None,
        neo4j_password: str = None,
        index_name: str = "clinical_guidelines_vector"
    ):
        """
        Initialize vector store with Neo4j.

        Args:
            neo4j_uri: Neo4j connection URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            index_name: Name of the vector index
        """
        self.neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_user = neo4j_user or os.getenv("NEO4J_USER", "neo4j")
        self.neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD", "password")
        self.index_name = index_name

        # Initialize embedding model
        logger.info("Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_dimension = 384  # all-MiniLM-L6-v2 dimension
        logger.info("✓ Embedding model loaded")

        # Initialize Neo4j connection
        try:
            self.driver = GraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.neo4j_user, self.neo4j_password)
            )
            logger.info(f"✓ Connected to Neo4j at {self.neo4j_uri}")
            self._setup_vector_index()
            doc_count = self._get_document_count()
            logger.info(f"✓ Vector store initialized: {doc_count} documents")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise

    def _setup_vector_index(self):
        """Create vector index for similarity search if it doesn't exist"""
        with self.driver.session() as session:
            # Check if index exists
            result = session.run(
                "SHOW INDEXES YIELD name WHERE name = $index_name RETURN count(*) as count",
                index_name=self.index_name
            )
            exists = result.single()["count"] > 0
            
            if not exists:
                # Create vector index (Neo4j 5.13+)
                session.run(f"""
                    CREATE VECTOR INDEX {self.index_name} IF NOT EXISTS
                    FOR (g:Guideline)
                    ON g.embedding
                    OPTIONS {{
                        indexConfig: {{
                            `vector.dimensions`: {self.embedding_dimension},
                            `vector.similarity_function`: 'cosine'
                        }}
                    }}
                """)
                logger.info(f"✓ Created vector index: {self.index_name}")
            else:
                logger.info(f"✓ Vector index exists: {self.index_name}")

    def _get_document_count(self) -> int:
        """Get total number of guideline documents"""
        with self.driver.session() as session:
            result = session.run("MATCH (g:Guideline) RETURN count(g) as count")
            return result.single()["count"]

    def add_guidelines(self, guidelines: List[Dict[str, Any]]) -> None:
        """
        Add clinical guidelines to vector store.

        Args:
            guidelines: List of guideline rule dictionaries
        """
        with self.driver.session() as session:
            for guideline in guidelines:
                # Create searchable text
                text = f"""
                Guideline: {guideline.get('name', '')}
                Source: {guideline.get('source', '')}
                Description: {guideline.get('description', '')}
                Treatment: {guideline.get('recommended_treatment', '')}
                Evidence Level: {guideline.get('evidence_level', '')}
                Intent: {guideline.get('treatment_intent', '')}
                """

                # Generate embedding
                embedding = self.embedding_model.encode(text.strip()).tolist()

                # Create or update guideline node with embedding
                session.run("""
                    MERGE (g:Guideline {rule_id: $rule_id})
                    SET g.name = $name,
                        g.source = $source,
                        g.description = $description,
                        g.treatment = $treatment,
                        g.evidence_level = $evidence_level,
                        g.treatment_intent = $treatment_intent,
                        g.document = $document,
                        g.embedding = $embedding,
                        g.updated_at = datetime()
                """, 
                    rule_id=guideline.get('rule_id', f"rule_{id(guideline)}"),
                    name=guideline.get('name', ''),
                    source=guideline.get('source', ''),
                    description=guideline.get('description', ''),
                    treatment=guideline.get('recommended_treatment', ''),
                    evidence_level=guideline.get('evidence_level', ''),
                    treatment_intent=guideline.get('treatment_intent', ''),
                    document=text.strip(),
                    embedding=embedding
                )

        logger.info(f"✓ Added {len(guidelines)} guidelines to Neo4j vector store")


    def search_guidelines(
        self,
        query: str,
        n_results: int = 5,
        filter_metadata: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant guidelines using semantic similarity.

        Args:
            query: Search query
            n_results: Number of results to return
            filter_metadata: Optional metadata filters (e.g., {"treatment": "Chemotherapy"})

        Returns:
            List of matching guidelines with scores
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query).tolist()

        # Build filter clause
        where_clause = ""
        params = {"query_embedding": query_embedding, "limit": n_results}
        
        if filter_metadata:
            conditions = []
            for key, value in filter_metadata.items():
                param_name = f"filter_{key}"
                conditions.append(f"g.{key} = ${param_name}")
                params[param_name] = value
            where_clause = "WHERE " + " AND ".join(conditions)

        # Execute vector similarity search
        with self.driver.session() as session:
            result = session.run(f"""
                CALL db.index.vector.queryNodes($index_name, $limit, $query_embedding)
                YIELD node, score
                {where_clause}
                RETURN node.rule_id as rule_id,
                       node.document as document,
                       node.treatment as treatment,
                       node.evidence_level as evidence_level,
                       node.source as source,
                       score as similarity_score
                ORDER BY score DESC
            """, index_name=self.index_name, **params)

            formatted_results = []
            for record in result:
                formatted_results.append({
                    "rule_id": record["rule_id"],
                    "document": record["document"],
                    "metadata": {
                        "treatment": record["treatment"],
                        "evidence_level": record["evidence_level"],
                        "source": record["source"]
                    },
                    "similarity_score": record["similarity_score"]
                })

            return formatted_results


    def find_similar_cases(
        self,
        patient_description: str,
        n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find similar patient cases or guidelines.

        Args:
            patient_description: Natural language patient description
            n_results: Number of results to return

        Returns:
            List of similar cases/guidelines
        """
        return self.search_guidelines(patient_description, n_results)

    def get_treatment_context(self, treatment_type: str) -> List[Dict[str, Any]]:
        """
        Get all guidelines related to a specific treatment.

        Args:
            treatment_type: Treatment name (e.g., "Chemotherapy")

        Returns:
            List of relevant guidelines
        """
        return self.search_guidelines(
            f"Treatment with {treatment_type}",
            n_results=10,
            filter_metadata={"treatment": treatment_type}
        )

    def clear_collection(self) -> None:
        """Clear all guideline documents from the vector store"""
        with self.driver.session() as session:
            session.run("MATCH (g:Guideline) DETACH DELETE g")
        logger.info("✓ Vector store cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        doc_count = self._get_document_count()
        return {
            "index_name": self.index_name,
            "total_documents": doc_count,
            "neo4j_uri": self.neo4j_uri,
            "embedding_model": "all-MiniLM-L6-v2",
            "embedding_dimension": self.embedding_dimension
        }

    def close(self):
        """Close Neo4j driver connection"""
        if self.driver:
            self.driver.close()
            logger.info("✓ Neo4j connection closed")



# Test function
if __name__ == "__main__":
    print("Testing Vector Store Integration")
    print("=" * 80)

    # Initialize
    vector_store = LUCADAVectorStore()

    # Sample guidelines
    test_guidelines = [
        {
            "rule_id": "R1",
            "name": "ChemoRule001_AdvancedNSCLC",
            "source": "NICE Lung Cancer 2011 - CG121",
            "description": "Offer chemotherapy to patients with stage III or IV NSCLC and good performance status",
            "recommended_treatment": "Chemotherapy",
            "evidence_level": "Grade A",
            "treatment_intent": "Palliative"
        },
        {
            "rule_id": "R2",
            "name": "SurgeryRule001_EarlyStageNSCLC",
            "source": "NICE Lung Cancer 2011 - CG121",
            "description": "Offer surgery to patients with stage I-II NSCLC and good performance status",
            "recommended_treatment": "Surgery",
            "evidence_level": "Grade A",
            "treatment_intent": "Curative"
        }
    ]

    # Add guidelines
    vector_store.add_guidelines(test_guidelines)

    # Test search
    results = vector_store.search_guidelines(
        "What treatment for early stage lung cancer?",
        n_results=2
    )

    print(f"\n✓ Found {len(results)} results:")
    for result in results:
        print(f"\n  Rule: {result['rule_id']}")
        print(f"  Similarity: {result['similarity_score']:.3f}")
        print(f"  Metadata: {result['metadata']}")

    # Get stats
    stats = vector_store.get_stats()
    print(f"\n✓ Vector Store Stats:")
    print(f"  Documents: {stats['total_documents']}")
    print(f"  Database: {stats['neo4j_uri']}")
    print(f"  Index: {stats['index_name']}")
    
    # Clean up
    vector_store.close()
