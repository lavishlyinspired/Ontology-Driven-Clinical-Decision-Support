#!/usr/bin/env python3
"""
Frontend Conversation Testing Automation Script
Automates testing of LCA Assistant conversations through the frontend API
"""

import asyncio
import aiohttp
import json
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import uuid

@dataclass
class ConversationTest:
    """Represents a single conversation test"""
    test_id: str
    name: str
    description: str
    conversation_flow: List[Dict[str, str]]
    expected_keywords: List[str] = None
    session_id: str = None
    
    def __post_init__(self):
        if self.session_id is None:
            self.session_id = f"test_{self.test_id}_{uuid.uuid4().hex[:8]}"
        if self.expected_keywords is None:
            self.expected_keywords = []

class FrontendConversationTester:
    """Automated testing for LCA frontend conversations"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results = []
    
    async def run_conversation_test(self, test: ConversationTest) -> Dict[str, Any]:
        """Run a single conversation test"""
        print(f"\n🧪 Running Test: {test.name}")
        print(f"   Description: {test.description}")
        
        results = {
            "test_id": test.test_id,
            "name": test.name,
            "status": "passed",
            "messages": [],
            "duration": 0,
            "errors": []
        }
        
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            try:
                for i, turn in enumerate(test.conversation_flow):
                    message_start = time.time()
                    
                    # Send message to chat endpoint
                    url = f"{self.base_url}/chat/stream"
                    payload = {
                        "message": turn["user"],
                        "session_id": test.session_id,
                        "use_enhanced_features": True
                    }
                    
                    print(f"   👤 User: {turn['user'][:100]}...")
                    
                    async with session.post(url, json=payload) as response:
                        if response.status == 200:
                            # For SSE endpoints, we'll get streaming data
                            response_text = ""
                            async for line in response.content:
                                line_str = line.decode('utf-8').strip()
                                if line_str.startswith('data: ') and line_str != 'data: [DONE]':
                                    try:
                                        data = json.loads(line_str[6:])  # Remove 'data: '
                                        if 'content' in data:
                                            response_text += data['content']
                                    except json.JSONDecodeError:
                                        continue
                            
                            message_duration = time.time() - message_start
                            
                            results["messages"].append({
                                "turn": i + 1,
                                "user_message": turn["user"],
                                "assistant_response": response_text,
                                "duration": message_duration,
                                "status": "success"
                            })
                            
                            print(f"   🤖 Assistant: {response_text[:100]}...")
                            print(f"   ⏱️  Response Time: {message_duration:.2f}s")
                            
                            # Check for expected keywords if specified
                            if test.expected_keywords and i == len(test.conversation_flow) - 1:
                                missing_keywords = []
                                for keyword in test.expected_keywords:
                                    if keyword.lower() not in response_text.lower():
                                        missing_keywords.append(keyword)
                                
                                if missing_keywords:
                                    results["errors"].append(f"Missing expected keywords: {missing_keywords}")
                                    results["status"] = "failed"
                            
                            # Small delay between messages
                            await asyncio.sleep(1)
                        
                        else:
                            error_msg = f"HTTP {response.status} for message {i+1}"
                            results["errors"].append(error_msg)
                            results["status"] = "failed"
                            print(f"   ❌ Error: {error_msg}")
                
                # Test follow-up suggestions if available
                await self._test_followup_suggestions(session, test.session_id, results)
                
            except Exception as e:
                results["errors"].append(f"Exception: {str(e)}")
                results["status"] = "failed"
                print(f"   ❌ Exception: {e}")
        
        results["duration"] = time.time() - start_time
        self.results.append(results)
        
        status_emoji = "✅" if results["status"] == "passed" else "❌"
        print(f"   {status_emoji} Test {results['status'].upper()} in {results['duration']:.2f}s")
        
        return results
    
    async def _test_followup_suggestions(self, session, session_id: str, results: Dict):
        """Test follow-up suggestion endpoint"""
        try:
            url = f"{self.base_url}/chat/follow-up/{session_id}"
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    suggestions_count = data.get("count", 0)
                    results["follow_up_count"] = suggestions_count
                    print(f"   💡 Follow-up suggestions: {suggestions_count}")
                else:
                    results["errors"].append(f"Follow-up endpoint returned {response.status}")
        except Exception as e:
            results["errors"].append(f"Follow-up test error: {str(e)}")

# Test Cases Definitions
CONVERSATION_TESTS = [
    ConversationTest(
        test_id="CONV_001",
        name="Basic Greeting and Introduction",
        description="Test basic conversational flow and system introduction",
        conversation_flow=[
            {"user": "Hello, I'm Dr. Smith. Can you help me with a lung cancer case?"}
        ],
        expected_keywords=["lung cancer", "assistant", "help", "analysis"]
    ),
    
    ConversationTest(
        test_id="CONV_002", 
        name="Simple Patient Case",
        description="Test processing of basic patient information",
        conversation_flow=[
            {"user": "I have a 65-year-old male patient with a 2cm lung nodule."},
            {"user": "The nodule is in the right upper lobe and the patient is a smoker."},
            {"user": "What staging workup would you recommend?"}
        ],
        expected_keywords=["staging", "workup", "CT", "PET", "biopsy"]
    ),
    
    ConversationTest(
        test_id="CONV_003",
        name="Enhanced Memory Test",
        description="Test conversation memory and context retention",
        conversation_flow=[
            {"user": "Patient is a 58-year-old female with chest pain."},
            {"user": "She has no smoking history but has family history of lung cancer."},
            {"user": "What's her risk profile based on what I told you?"}
        ],
        expected_keywords=["58", "female", "non-smoker", "family history"]
    ),
    
    ConversationTest(
        test_id="CONV_004",
        name="Complex Medical Case",
        description="Test handling of complex medical information",
        conversation_flow=[
            {"user": "72-year-old male, 50 pack-year smoking history, presenting with hemoptysis and weight loss."},
            {"user": "CT shows 4.5cm mass in LUL with mediastinal adenopathy and liver lesions."},
            {"user": "Biopsy confirms squamous cell carcinoma. What's the staging and treatment approach?"}
        ],
        expected_keywords=["stage IV", "squamous cell", "palliative", "immunotherapy"]
    ),
    
    ConversationTest(
        test_id="CONV_005",
        name="Molecular Testing Discussion",
        description="Test biomarker and molecular testing knowledge",
        conversation_flow=[
            {"user": "Patient has adenocarcinoma with EGFR exon 19 deletion."},
            {"user": "PD-L1 expression is 75%. What treatment would you recommend?"}
        ],
        expected_keywords=["EGFR", "TKI", "osimertinib", "targeted therapy"]
    ),
    
    ConversationTest(
        test_id="CONV_006",
        name="Follow-up Generation Test",
        description="Test intelligent follow-up question generation",
        conversation_flow=[
            {"user": "Patient has ground-glass opacities on chest CT."}
        ],
        expected_keywords=["size", "location", "symptoms", "history"]
    ),
    
    ConversationTest(
        test_id="CONV_007",
        name="TNM Staging Query",
        description="Test TNM staging knowledge and application",
        conversation_flow=[
            {"user": "Help me stage a patient with 3.2cm primary tumor, mediastinal nodes, no metastases."},
            {"user": "The mediastinal nodes are ipsilateral."}
        ],
        expected_keywords=["T2a", "N2", "M0", "stage IIIA"]
    ),
    
    ConversationTest(
        test_id="CONV_008",
        name="Treatment Guidelines",
        description="Test knowledge of treatment guidelines and protocols",
        conversation_flow=[
            {"user": "What are current NCCN guidelines for stage IIIA lung cancer treatment?"}
        ],
        expected_keywords=["NCCN", "multimodal", "neoadjuvant", "surgery", "radiation"]
    ),
    
    ConversationTest(
        test_id="CONV_009",
        name="Error Handling Test",
        description="Test handling of unclear or contradictory information",
        conversation_flow=[
            {"user": "Patient has T1N0M0 cancer but also has brain metastases."}
        ],
        expected_keywords=["contradict", "clarif", "brain metastases", "M1"]
    ),
    
    ConversationTest(
        test_id="CONV_010",
        name="Multi-turn Complex Case",
        description="Test extended conversation with complex case building",
        conversation_flow=[
            {"user": "New patient consult for lung cancer."},
            {"user": "64-year-old never-smoker female, Asian ethnicity."},
            {"user": "Presents with persistent cough and fatigue."},
            {"user": "CT shows 2.8cm adenocarcinoma, no nodes or mets."},
            {"user": "EGFR testing shows exon 21 L858R mutation."},
            {"user": "What's your treatment recommendation?"}
        ],
        expected_keywords=["never-smoker", "Asian", "EGFR", "L858R", "TKI", "first-line"]
    )
]

async def run_all_tests():
    """Run all conversation tests"""
    print("🚀 Starting LCA Frontend Conversation Tests")
    print("=" * 60)
    
    tester = FrontendConversationTester()
    
    # Run all tests
    for test in CONVERSATION_TESTS:
        await tester.run_conversation_test(test)
        await asyncio.sleep(2)  # Delay between tests
    
    # Generate summary report
    generate_summary_report(tester.results)

def generate_summary_report(results: List[Dict[str, Any]]):
    """Generate summary report of test results"""
    passed = len([r for r in results if r["status"] == "passed"])
    failed = len([r for r in results if r["status"] == "failed"])
    total = len(results)
    
    total_duration = sum(r["duration"] for r in results)
    avg_response_time = sum([
        sum(msg["duration"] for msg in r["messages"]) / len(r["messages"]) 
        for r in results if r["messages"]
    ]) / len([r for r in results if r["messages"]])
    
    print("\n" + "=" * 60)
    print("📊 LCA Frontend Conversation Test Summary")
    print("=" * 60)
    
    print(f"\n🎯 Results:")
    print(f"   ✅ Passed: {passed}/{total} ({(passed/total)*100:.1f}%)")
    print(f"   ❌ Failed: {failed}/{total}")
    print(f"   ⏱️  Total Duration: {total_duration:.2f}s")
    print(f"   📈 Avg Response Time: {avg_response_time:.2f}s")
    
    # Failed tests
    failed_tests = [r for r in results if r["status"] == "failed"]
    if failed_tests:
        print(f"\n❌ Failed Tests:")
        for test in failed_tests:
            print(f"   • {test['name']}")
            for error in test["errors"]:
                print(f"     - {error}")
    
    # Performance analysis
    print(f"\n⚡ Performance Analysis:")
    slow_tests = [r for r in results if r["duration"] > 30]
    if slow_tests:
        print(f"   ⚠️ Slow Tests (>30s):")
        for test in slow_tests:
            print(f"     • {test['name']}: {test['duration']:.2f}s")
    else:
        print(f"   ✅ All tests completed in reasonable time")
    
    # Follow-up analysis
    followup_counts = [r.get("follow_up_count", 0) for r in results if "follow_up_count" in r]
    if followup_counts:
        avg_followups = sum(followup_counts) / len(followup_counts)
        print(f"   💡 Avg Follow-up Questions: {avg_followups:.1f}")
    
    print("\n" + "=" * 60)

async def run_single_test(test_id: str):
    """Run a single test by ID"""
    test = next((t for t in CONVERSATION_TESTS if t.test_id == test_id), None)
    if not test:
        print(f"❌ Test {test_id} not found")
        return
    
    tester = FrontendConversationTester()
    result = await tester.run_conversation_test(test)
    
    print(f"\n📋 Single Test Report:")
    print(f"   Test: {result['name']}")
    print(f"   Status: {result['status']}")
    print(f"   Duration: {result['duration']:.2f}s")
    if result['errors']:
        print(f"   Errors: {result['errors']}")

async def main():
    """Main entry point with argument parsing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="LCA Frontend Conversation Tester")
    parser.add_argument("--base-url", default="http://localhost:8000",
                       help="Base URL for LCA API")
    parser.add_argument("--test-id", help="Run single test by ID")
    parser.add_argument("--list", action="store_true", help="List all available tests")
    
    args = parser.parse_args()
    
    if args.list:
        print("📋 Available Tests:")
        for test in CONVERSATION_TESTS:
            print(f"   {test.test_id}: {test.name}")
            print(f"      {test.description}")
        return
    
    # Update base URL if provided
    FrontendConversationTester.__init__.__defaults__ = (args.base_url,)
    
    if args.test_id:
        await run_single_test(args.test_id)
    else:
        await run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())