"""
RAG (Retrieval-Augmented Generation) Enhancement Service

Enhances guideline retrieval using vector embeddings and semantic search.
Provides context-aware question answering and similar case retrieval.
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pydantic import BaseModel
import numpy as np


class GuidelineChunk(BaseModel):
    """A chunk of guideline text with embeddings."""
    chunk_id: str
    guideline_id: str
    guideline_version: str
    section: str
    content: str
    
    # Metadata
    page_number: Optional[int] = None
    evidence_level: Optional[str] = None  # 1, 2A, 2B, 3
    recommendation_category: Optional[str] = None  # Preferred, Alternative, etc.
    
    # Vector embedding
    embedding: Optional[List[float]] = None
    
    # Source tracking
    source_url: Optional[str] = None
    last_updated: datetime


class RetrievalResult(BaseModel):
    """Result from vector search."""
    chunk: GuidelineChunk
    score: float
    relevance_explanation: Optional[str] = None


class SimilarCase(BaseModel):
    """A similar historical case."""
    case_id: str
    patient_id: str
    similarity_score: float
    
    # Patient characteristics
    age: int
    stage: str
    histology: str
    biomarkers: Dict[str, Any]
    
    # Treatment and outcome
    treatment: str
    outcome: str
    survival_months: Optional[float] = None
    
    # Match reasons
    match_factors: List[str]


class RAGService:
    """Service for RAG-enhanced guideline retrieval."""
    
    def __init__(self):
        """Initialize RAG service."""
        from ..db.vector_store import vector_store
        
        self.vector_store = vector_store
        self._embedding_model = None
        self._guideline_chunks: List[GuidelineChunk] = []
        self._case_database: List[Dict[str, Any]] = []
    
    def initialize_embeddings(self):
        """Initialize embedding model (lazy loading)."""
        if self._embedding_model is None:
            try:
                # Use sentence-transformers for embeddings
                from sentence_transformers import SentenceTransformer
                
                # Medical domain-specific model
                self._embedding_model = SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO')
                print("✅ Loaded medical embedding model: S-PubMedBert-MS-MARCO")
            except ImportError:
                # Fallback to basic model
                from sentence_transformers import SentenceTransformer
                self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                print("⚠️ Using fallback embedding model: all-MiniLM-L6-v2")
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for text."""
        self.initialize_embeddings()
        embedding = self._embedding_model.encode(text)
        return embedding.tolist()
    
    def index_guideline(
        self,
        guideline_id: str,
        guideline_version: str,
        guideline_text: str,
        metadata: Dict[str, Any]
    ) -> int:
        """
        Index a clinical guideline for retrieval.
        
        Chunks the guideline and creates embeddings.
        
        Returns:
            Number of chunks created
        """
        chunks = self._chunk_guideline(guideline_text, metadata)
        
        for i, chunk_data in enumerate(chunks):
            chunk_id = f"{guideline_id}_v{guideline_version}_{i:04d}"
            
            # Create embedding
            embedding = self.embed_text(chunk_data['content'])
            
            chunk = GuidelineChunk(
                chunk_id=chunk_id,
                guideline_id=guideline_id,
                guideline_version=guideline_version,
                section=chunk_data.get('section', 'Unknown'),
                content=chunk_data['content'],
                page_number=chunk_data.get('page'),
                evidence_level=chunk_data.get('evidence_level'),
                recommendation_category=chunk_data.get('category'),
                embedding=embedding,
                source_url=metadata.get('source_url'),
                last_updated=datetime.now()
            )
            
            self._guideline_chunks.append(chunk)
            
            # Also store in vector store
            self.vector_store.add_embedding(
                id=chunk_id,
                embedding=embedding,
                metadata={
                    'guideline_id': guideline_id,
                    'version': guideline_version,
                    'section': chunk.section,
                    'content': chunk.content
                }
            )
        
        print(f"📚 Indexed {len(chunks)} chunks from {guideline_id} v{guideline_version}")
        
        return len(chunks)
    
    def _chunk_guideline(
        self,
        text: str,
        metadata: Dict[str, Any],
        chunk_size: int = 500,
        overlap: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Chunk guideline text intelligently.
        
        Uses section headers and paragraph boundaries.
        """
        chunks = []
        
        # Split by sections (simplified - in reality would parse structure)
        sections = text.split('\n\n')
        
        current_chunk = []
        current_length = 0
        current_section = "Introduction"
        
        for section in sections:
            # Detect section headers
            if section.isupper() or section.startswith('#'):
                current_section = section.strip('#').strip()
                continue
            
            words = section.split()
            
            if current_length + len(words) > chunk_size:
                # Save current chunk
                chunks.append({
                    'content': ' '.join(current_chunk),
                    'section': current_section,
                    'evidence_level': metadata.get('evidence_level'),
                    'category': metadata.get('category')
                })
                
                # Start new chunk with overlap
                overlap_words = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                current_chunk = overlap_words + words
                current_length = len(current_chunk)
            else:
                current_chunk.extend(words)
                current_length += len(words)
        
        # Add final chunk
        if current_chunk:
            chunks.append({
                'content': ' '.join(current_chunk),
                'section': current_section,
                'evidence_level': metadata.get('evidence_level'),
                'category': metadata.get('category')
            })
        
        return chunks
    
    def retrieve_relevant_guidelines(
        self,
        query: str,
        patient_context: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
        min_score: float = 0.6
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant guideline chunks using semantic search.
        
        Args:
            query: Search query (e.g., "treatment for stage IIIB NSCLC with EGFR mutation")
            patient_context: Additional patient context to refine search
            top_k: Number of results to return
            min_score: Minimum similarity score
        
        Returns:
            List of relevant guideline chunks with scores
        """
        # Enhance query with patient context
        enhanced_query = self._enhance_query(query, patient_context)
        
        # Generate query embedding
        query_embedding = self.embed_text(enhanced_query)
        
        # Search in chunks
        results = []
        for chunk in self._guideline_chunks:
            if chunk.embedding is None:
                continue
            
            # Calculate cosine similarity
            score = self._cosine_similarity(query_embedding, chunk.embedding)
            
            if score >= min_score:
                results.append(RetrievalResult(
                    chunk=chunk,
                    score=score,
                    relevance_explanation=self._explain_relevance(query, chunk, score)
                ))
        
        # Sort by score
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results[:top_k]
    
    def _enhance_query(
        self,
        query: str,
        patient_context: Optional[Dict[str, Any]]
    ) -> str:
        """Enhance query with patient context."""
        if not patient_context:
            return query
        
        enhancements = []
        
        if 'stage' in patient_context:
            enhancements.append(f"stage {patient_context['stage']}")
        
        if 'histology' in patient_context:
            enhancements.append(patient_context['histology'])
        
        if 'biomarkers' in patient_context:
            biomarkers = patient_context['biomarkers']
            if biomarkers:
                biomarker_str = ', '.join([f"{k}={v}" for k, v in biomarkers.items()])
                enhancements.append(f"biomarkers: {biomarker_str}")
        
        if 'ps' in patient_context:
            enhancements.append(f"PS {patient_context['ps']}")
        
        if enhancements:
            return f"{query} ({'; '.join(enhancements)})"
        
        return query
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    def _explain_relevance(
        self,
        query: str,
        chunk: GuidelineChunk,
        score: float
    ) -> str:
        """Generate explanation for why this chunk is relevant."""
        query_words = set(query.lower().split())
        chunk_words = set(chunk.content.lower().split())
        
        overlap = query_words & chunk_words
        
        explanation_parts = [f"Similarity: {score:.1%}"]
        
        if overlap:
            explanation_parts.append(f"Matched terms: {', '.join(list(overlap)[:5])}")
        
        if chunk.evidence_level:
            explanation_parts.append(f"Evidence level: {chunk.evidence_level}")
        
        if chunk.recommendation_category:
            explanation_parts.append(f"Category: {chunk.recommendation_category}")
        
        return ". ".join(explanation_parts)
    
    def answer_question(
        self,
        question: str,
        patient_context: Optional[Dict[str, Any]] = None,
        use_llm: bool = True
    ) -> Dict[str, Any]:
        """
        Answer a clinical question using RAG.
        
        Retrieves relevant guidelines and generates an answer.
        """
        # Retrieve relevant context
        relevant_chunks = self.retrieve_relevant_guidelines(
            query=question,
            patient_context=patient_context,
            top_k=3
        )
        
        if not relevant_chunks:
            return {
                'answer': 'No relevant guidelines found for this question.',
                'confidence': 0.0,
                'sources': []
            }
        
        # Prepare context for LLM
        context = '\n\n'.join([
            f"[Source: {r.chunk.guideline_id} v{r.chunk.guideline_version}, Section: {r.chunk.section}]\n{r.chunk.content}"
            for r in relevant_chunks
        ])
        
        if use_llm:
            # Use LLM to generate answer
            from .llm_extractor import llm_extractor
            
            prompt = f"""Based on the following clinical guidelines, answer the question.

Question: {question}

Relevant Guidelines:
{context}

Provide a concise, evidence-based answer with specific recommendations."""
            
            answer = llm_extractor.ask_question(prompt)
        else:
            # Simple extraction (no LLM)
            answer = f"Based on {len(relevant_chunks)} relevant guideline sections:\n\n"
            answer += relevant_chunks[0].chunk.content[:300] + "..."
        
        # Calculate confidence based on retrieval scores
        avg_score = sum(r.score for r in relevant_chunks) / len(relevant_chunks)
        
        return {
            'answer': answer,
            'confidence': avg_score,
            'sources': [
                {
                    'guideline_id': r.chunk.guideline_id,
                    'version': r.chunk.guideline_version,
                    'section': r.chunk.section,
                    'score': r.score,
                    'url': r.chunk.source_url
                }
                for r in relevant_chunks
            ],
            'retrieved_chunks': len(relevant_chunks)
        }
    
    def find_similar_cases(
        self,
        patient_data: Dict[str, Any],
        top_k: int = 5,
        min_similarity: float = 0.7
    ) -> List[SimilarCase]:
        """
        Find similar historical cases using patient embeddings.
        
        Helps clinicians see outcomes for similar patients.
        """
        # Create patient embedding
        patient_description = self._create_patient_description(patient_data)
        patient_embedding = self.embed_text(patient_description)
        
        # Search case database
        similar_cases = []
        
        for case in self._case_database:
            case_embedding = case.get('embedding')
            if case_embedding is None:
                continue
            
            similarity = self._cosine_similarity(patient_embedding, case_embedding)
            
            if similarity >= min_similarity:
                match_factors = self._identify_match_factors(patient_data, case)
                
                similar_cases.append(SimilarCase(
                    case_id=case['case_id'],
                    patient_id=case['patient_id'],
                    similarity_score=similarity,
                    age=case['age'],
                    stage=case['stage'],
                    histology=case['histology'],
                    biomarkers=case.get('biomarkers', {}),
                    treatment=case['treatment'],
                    outcome=case['outcome'],
                    survival_months=case.get('survival_months'),
                    match_factors=match_factors
                ))
        
        # Sort by similarity
        similar_cases.sort(key=lambda x: x.similarity_score, reverse=True)
        
        return similar_cases[:top_k]
    
    def _create_patient_description(self, patient_data: Dict[str, Any]) -> str:
        """Create textual description of patient for embedding."""
        parts = []
        
        if 'age' in patient_data:
            parts.append(f"{patient_data['age']} years old")
        
        if 'stage' in patient_data:
            parts.append(f"stage {patient_data['stage']}")
        
        if 'histology' in patient_data:
            parts.append(patient_data['histology'])
        
        if 'biomarkers' in patient_data:
            biomarkers = patient_data['biomarkers']
            if biomarkers:
                bio_str = ', '.join([f"{k} {v}" for k, v in biomarkers.items()])
                parts.append(f"biomarkers: {bio_str}")
        
        if 'ps' in patient_data:
            parts.append(f"performance status {patient_data['ps']}")
        
        if 'comorbidities' in patient_data:
            comorbidities = patient_data['comorbidities']
            if comorbidities:
                parts.append(f"comorbidities: {', '.join(comorbidities)}")
        
        return '. '.join(parts)
    
    def _identify_match_factors(
        self,
        patient_data: Dict[str, Any],
        case: Dict[str, Any]
    ) -> List[str]:
        """Identify what makes these cases similar."""
        factors = []
        
        # Stage match
        if patient_data.get('stage') == case.get('stage'):
            factors.append(f"Same stage ({case['stage']})")
        
        # Histology match
        if patient_data.get('histology') == case.get('histology'):
            factors.append(f"Same histology ({case['histology']})")
        
        # Biomarker match
        patient_bio = patient_data.get('biomarkers', {})
        case_bio = case.get('biomarkers', {})
        
        common_biomarkers = set(patient_bio.keys()) & set(case_bio.keys())
        if common_biomarkers:
            factors.append(f"Shared biomarkers: {', '.join(common_biomarkers)}")
        
        # Age similarity
        if 'age' in patient_data and 'age' in case:
            age_diff = abs(patient_data['age'] - case['age'])
            if age_diff < 10:
                factors.append(f"Similar age (within {age_diff} years)")
        
        return factors
    
    def index_case(
        self,
        case_id: str,
        patient_data: Dict[str, Any],
        treatment: str,
        outcome: str,
        survival_months: Optional[float] = None
    ):
        """Index a historical case for similarity search."""
        # Create patient description and embedding
        description = self._create_patient_description(patient_data)
        embedding = self.embed_text(description)
        
        case_record = {
            'case_id': case_id,
            'patient_id': patient_data.get('patient_id', 'unknown'),
            'age': patient_data.get('age', 65),
            'stage': patient_data.get('stage', 'Unknown'),
            'histology': patient_data.get('histology', 'Unknown'),
            'biomarkers': patient_data.get('biomarkers', {}),
            'treatment': treatment,
            'outcome': outcome,
            'survival_months': survival_months,
            'embedding': embedding,
            'indexed_at': datetime.now()
        }
        
        self._case_database.append(case_record)
        
        print(f"💾 Indexed case {case_id} for similarity search")


# Global RAG service instance
rag_service = RAGService()
