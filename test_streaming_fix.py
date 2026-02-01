"""
Test script to debug the streaming issue
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


async def test_streaming():
    """Test the actual streaming functionality"""
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
    print("Testing chat_stream functionality")
    print("=" * 80)
    print(f"Input message: {message}")
    print(f"Session ID: {session_id}")
    print()

    print("SSE Events received:")
    print("-" * 80)

    event_count = 0
    event_types = {}
    content_received = False

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

                        # Track event types
                        event_types[event_type] = event_types.get(event_type, 0) + 1

                        # Check if this is content
                        if event_type in ["recommendation", "text", "clinical_summary", "treatment_plan"]:
                            content_received = True
                            print(f"  [{event_count}] {event_type}: {data.get('content', '')[:100]}...")
                        elif event_type == "error":
                            print(f"  [{event_count}] ERROR: {data.get('content', '')}")
                        else:
                            print(f"  [{event_count}] {event_type}: {data.get('content', '')}")
                    except json.JSONDecodeError as e:
                        print(f"  [{event_count}] JSON decode error: {e}")
                        print(f"     Raw data: {data_str[:200]}")

    except Exception as e:
        print(f"\nERROR during streaming: {e}")
        import traceback
        traceback.print_exc()

    print("-" * 80)
    print(f"\nTotal events: {event_count}")
    print(f"Event types:")
    for event_type, count in event_types.items():
        print(f"  {event_type}: {count}")
    print(f"\nContent received: {content_received}")

    if not content_received:
        print("\nWARNING: No content events received!")
        print("This would cause 'No Response Generated' in the frontend")

    lca_service.close()


if __name__ == "__main__":
    asyncio.run(test_streaming())
