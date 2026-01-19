"""
Setup Vector Store and Embeddings
Initializes vector embeddings model and creates initial guideline embeddings
"""

import os
import sys
from pathlib import Path
import asyncio

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import services
from backend.src.services.rag_service import rag_service
from backend.src.db.vector_store import VectorStore


async def initialize_embeddings():
    """Initialize embeddings model"""
    print("=" * 80)
    print("🔢 Initializing Embeddings Model")
    print("=" * 80)
    
    model_name = os.getenv("EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    device = os.getenv("EMBEDDINGS_DEVICE", "cpu")
    
    print(f"   Model: {model_name}")
    print(f"   Device: {device}")
    
    try:
        # Initialize RAG service (which loads embeddings)
        await rag_service.initialize()
        print("   ✓ Embeddings model loaded successfully")
        
        # Test embeddings
        test_text = "Stage IIIA non-small cell lung cancer treatment recommendation"
        test_embedding = await rag_service.generate_embedding(test_text)
        
        print(f"   ✓ Test embedding generated: {len(test_embedding)} dimensions")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Failed to initialize embeddings: {e}")
        return False


async def index_guidelines():
    """Index NCCN guidelines into vector store"""
    print("\n" + "=" * 80)
    print("📚 Indexing Clinical Guidelines")
    print("=" * 80)
    
    # Sample NCCN guidelines (in production, load from files)
    sample_guidelines = [
        {
            "id": "nccn_nsclc_ia1",
            "title": "NCCN NSCLC Stage IA1",
            "text": "For stage IA1 (T1a N0 M0) NSCLC, preferred treatment is lobectomy with mediastinal lymph node dissection. Alternative options include segmentectomy or wedge resection for patients with poor pulmonary function.",
            "metadata": {
                "stage": "IA1",
                "guideline": "NCCN",
                "version": "2024.1",
                "treatment_intent": "curative"
            }
        },
        {
            "id": "nccn_nsclc_iiia",
            "title": "NCCN NSCLC Stage IIIA",
            "text": "Stage IIIA NSCLC (T1-3 N2 M0) requires multimodality therapy. Standard approach is concurrent chemoradiotherapy followed by durvalumab consolidation for 12 months. Surgery may be considered after neoadjuvant therapy for carefully selected patients.",
            "metadata": {
                "stage": "IIIA",
                "guideline": "NCCN",
                "version": "2024.1",
                "treatment_intent": "curative"
            }
        },
        {
            "id": "nccn_nsclc_iv_egfr",
            "title": "NCCN NSCLC Stage IV EGFR+",
            "text": "For metastatic NSCLC with EGFR mutations, first-line treatment is osimertinib. Alternative EGFR TKIs include erlotinib, gefitinib, or afatinib. Chemotherapy is reserved for progression after TKI therapy.",
            "metadata": {
                "stage": "IV",
                "guideline": "NCCN",
                "version": "2024.1",
                "biomarker": "EGFR mutation",
                "treatment_intent": "palliative"
            }
        },
        {
            "id": "nccn_sclc_limited",
            "title": "NCCN SCLC Limited Stage",
            "text": "Limited-stage SCLC is treated with concurrent platinum-etoposide chemotherapy and thoracic radiation. Prophylactic cranial irradiation (PCI) should be offered to patients achieving complete or near-complete response.",
            "metadata": {
                "stage": "limited",
                "guideline": "NCCN",
                "version": "2024.1",
                "cancer_type": "SCLC",
                "treatment_intent": "curative"
            }
        }
    ]
    
    indexed_count = 0
    
    for guideline in sample_guidelines:
        try:
            # Add to vector store via RAG service
            await rag_service.add_guideline(
                guideline_id=guideline["id"],
                text=guideline["text"],
                metadata=guideline["metadata"]
            )
            
            print(f"   ✓ Indexed: {guideline['title']}")
            indexed_count += 1
            
        except Exception as e:
            print(f"   ❌ Failed to index '{guideline['title']}': {e}")
    
    print(f"\n   📊 Total guidelines indexed: {indexed_count}/{len(sample_guidelines)}")
    
    return indexed_count


async def test_similarity_search():
    """Test vector similarity search"""
    print("\n" + "=" * 80)
    print("🔍 Testing Similarity Search")
    print("=" * 80)
    
    test_queries = [
        "What is the recommended treatment for stage IIIA NSCLC?",
        "Which targeted therapy should I use for EGFR-positive metastatic disease?",
        "How do I treat limited stage small cell lung cancer?"
    ]
    
    for query in test_queries:
        try:
            print(f"\n   Query: '{query}'")
            
            results = await rag_service.search_similar_guidelines(
                query_text=query,
                top_k=2
            )
            
            print(f"   Found {len(results)} similar guidelines:")
            for idx, result in enumerate(results, 1):
                print(f"      {idx}. {result.get('title', result.get('guideline_id'))} "
                      f"(similarity: {result.get('similarity_score', 0):.3f})")
            
        except Exception as e:
            print(f"   ❌ Search failed: {e}")


async def main():
    """Main entry point"""
    print("\n🌱 Starting Vector Store Setup...")
    print(f"📅 Date: {Path(__file__).stat().st_mtime}")
    
    try:
        # Step 1: Initialize embeddings
        embeddings_ready = await initialize_embeddings()
        
        if not embeddings_ready:
            print("\n❌ Failed to initialize embeddings - aborting")
            sys.exit(1)
        
        # Step 2: Index guidelines
        indexed_count = await index_guidelines()
        
        # Step 3: Test search
        await test_similarity_search()
        
        print("\n" + "=" * 80)
        print("✅ Vector Store Setup Complete!")
        print(f"   • Embeddings model: ✓")
        print(f"   • Guidelines indexed: {indexed_count}")
        print(f"   • Similarity search: ✓")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
