"""
Quick test to verify the fix works
"""

import asyncio
import sys
import os
from pathlib import Path
import json

# Fix Windows console encoding
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, 'strict')

# Set up paths
project_root = Path(__file__).parent
backend_root = project_root / "backend"
os.chdir(str(backend_root))
sys.path.insert(0, str(backend_root))

from src.services.lca_service import LungCancerAssistantService
from src.services.conversation_service import ConversationService


async def test_fix():
    """Test if the fix resolves the issue"""
    # Initialize services
    lca_service = LungCancerAssistantService(
        use_neo4j=False,
        use_vector_store=False,
        enable_advanced_workflow=False,
        enable_provenance=False
    )

    conversation_service = ConversationService(lca_service, enable_enhanced_features=False)

    # Test message
    message = "68M, stage IIIA adenocarcinoma, EGFR Ex19del+"
    session_id = "test_session"

    print("=" * 80)
    print("Testing the fix for: " + message)
    print("=" * 80)

    recommendation_received = False
    error_received = False
    event_count = 0

    try:
        async for chunk in conversation_service.chat_stream(
            session_id=session_id,
            message=message,
            use_enhanced_features=False
        ):
            event_count += 1

            # Parse SSE format
            if chunk.startswith("data: "):
                data_str = chunk[6:].strip()
                if data_str:
                    try:
                        data = json.loads(data_str)
                        event_type = data.get("type", "unknown")

                        # Check for recommendation or error
                        if event_type == "recommendation":
                            recommendation_received = True
                            content = data.get('content', '')
                            # Print first 200 chars to verify it's valid
                            safe_content = content.replace('\n', ' ')[:200]
                            print(f"✓ Recommendation received: {safe_content}...")
                        elif event_type == "error":
                            error_received = True
                            print(f"✗ Error received: {data.get('content', '')}")
                    except:
                        pass

    except Exception as e:
        print(f"✗ Exception during streaming: {e}")
        error_received = True

    print("\n" + "=" * 80)
    print("Test Results:")
    print("=" * 80)
    print(f"Total events: {event_count}")
    print(f"Recommendation received: {recommendation_received}")
    print(f"Error received: {error_received}")

    if recommendation_received and not error_received:
        print("\n✓✓✓ TEST PASSED - Fix is working! ✓✓✓")
    else:
        print("\n✗✗✗ TEST FAILED - Issue persists ✗✗✗")

    lca_service.close()


if __name__ == "__main__":
    asyncio.run(test_fix())
