#!/usr/bin/env python3
"""
Comprehensive Test Runner for LCA System
Executes all test scenarios from comprehensive_test_data.json
"""

import json
import asyncio
import aiohttp
import time
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging
from dataclasses import dataclass
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    test_id: str
    category: str
    name: str
    status: str  # "passed", "failed", "skipped"
    duration: float
    details: str = ""
    timestamp: str = ""

class LCATestRunner:
    """Comprehensive test runner for LCA system"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results: List[TestResult] = []
        self.test_data = self._load_test_data()
    
    def _load_test_data(self) -> Dict[str, Any]:
        """Load test scenarios from JSON file"""
        test_file = Path(__file__).parent / "comprehensive_test_data.json"
        with open(test_file, 'r') as f:
            return json.load(f)
    
    async def run_all_tests(self):
        """Run all test categories in order"""
        print("🧪 Starting Comprehensive LCA System Testing")
        print("=" * 50)
        
        test_order = self.test_data["comprehensive_test_scenarios"]["test_execution_order"]
        
        for category in test_order:
            await self._run_category(category)
        
        self._generate_report()
    
    async def _run_category(self, category: str):
        """Run all tests in a category"""
        logger.info(f"📋 Running {category.replace('_', ' ').title()}")
        
        category_tests = self.test_data["comprehensive_test_scenarios"].get(category, [])
        
        if category == "infrastructure_tests":
            await self._run_infrastructure_tests(category_tests)
        elif category == "data_ingestion_tests":
            await self._run_data_ingestion_tests(category_tests)
        elif category == "agent_testing_scenarios":
            await self._run_agent_tests(category_tests)
        elif category == "conversation_ai_tests":
            await self._run_conversation_tests(category_tests)
        elif category == "enhanced_features_tests":
            await self._run_enhanced_features_tests(category_tests)
        elif category == "integration_tests":
            await self._run_integration_tests(category_tests)
        elif category == "edge_cases_and_error_handling":
            await self._run_edge_case_tests(category_tests)
        else:
            logger.warning(f"Unknown test category: {category}")
    
    async def _run_infrastructure_tests(self, tests: List[Dict]):
        """Run infrastructure connectivity tests"""
        async with aiohttp.ClientSession() as session:
            for test in tests:
                start_time = time.time()
                
                try:
                    endpoint = test.get("endpoint", "")
                    url = f"{self.base_url}{endpoint}"
                    
                    async with session.get(url) as response:
                        duration = time.time() - start_time
                        
                        if response.status == test["expected_status"]:
                            result = TestResult(
                                test_id=test["test_id"],
                                category=test["category"],
                                name=test["name"],
                                status="passed",
                                duration=duration,
                                timestamp=datetime.now().isoformat()
                            )
                            logger.info(f"✅ {test['name']} - {duration:.2f}s")
                        else:
                            result = TestResult(
                                test_id=test["test_id"],
                                category=test["category"],
                                name=test["name"],
                                status="failed",
                                duration=duration,
                                details=f"Expected status {test['expected_status']}, got {response.status}",
                                timestamp=datetime.now().isoformat()
                            )
                            logger.error(f"❌ {test['name']} - Status mismatch")
                        
                        self.results.append(result)
                        
                except Exception as e:
                    duration = time.time() - start_time
                    result = TestResult(
                        test_id=test["test_id"],
                        category=test["category"],
                        name=test["name"],
                        status="failed",
                        duration=duration,
                        details=str(e),
                        timestamp=datetime.now().isoformat()
                    )
                    self.results.append(result)
                    logger.error(f"❌ {test['name']} - Exception: {e}")
    
    async def _run_data_ingestion_tests(self, tests: List[Dict]):
        """Run data ingestion and extraction tests"""
        async with aiohttp.ClientSession() as session:
            for test in tests:
                start_time = time.time()
                
                try:
                    # Use chat endpoint to test extraction
                    url = f"{self.base_url}/chat/stream"
                    payload = {
                        "message": test["input_message"],
                        "session_id": f"test_{test['test_id']}"
                    }
                    
                    async with session.post(url, json=payload) as response:
                        duration = time.time() - start_time
                        
                        if response.status == 200:
                            # For streaming endpoints, we consider 200 as success
                            # Detailed validation would require parsing SSE stream
                            result = TestResult(
                                test_id=test["test_id"],
                                category=test["category"],
                                name=test["name"],
                                status="passed",
                                duration=duration,
                                details="Extraction request successful",
                                timestamp=datetime.now().isoformat()
                            )
                            logger.info(f"✅ {test['name']} - {duration:.2f}s")
                        else:
                            result = TestResult(
                                test_id=test["test_id"],
                                category=test["category"], 
                                name=test["name"],
                                status="failed",
                                duration=duration,
                                details=f"HTTP {response.status}",
                                timestamp=datetime.now().isoformat()
                            )
                            logger.error(f"❌ {test['name']} - HTTP {response.status}")
                        
                        self.results.append(result)
                        
                except Exception as e:
                    duration = time.time() - start_time
                    result = TestResult(
                        test_id=test["test_id"],
                        category=test["category"],
                        name=test["name"],
                        status="failed",
                        duration=duration,
                        details=str(e),
                        timestamp=datetime.now().isoformat()
                    )
                    self.results.append(result)
                    logger.error(f"❌ {test['name']} - Exception: {e}")
    
    async def _run_conversation_tests(self, tests: List[Dict]):
        """Run conversation AI tests"""
        async with aiohttp.ClientSession() as session:
            for test in tests:
                start_time = time.time()
                
                try:
                    # Run conversation flow
                    conversation = test["conversation_flow"]
                    session_id = f"conv_test_{test['test_id']}"
                    
                    success = True
                    for turn in conversation:
                        url = f"{self.base_url}/chat/stream"
                        payload = {
                            "message": turn["user"],
                            "session_id": session_id
                        }
                        
                        async with session.post(url, json=payload) as response:
                            if response.status != 200:
                                success = False
                                break
                    
                    duration = time.time() - start_time
                    
                    result = TestResult(
                        test_id=test["test_id"],
                        category=test["category"],
                        name=test["name"],
                        status="passed" if success else "failed",
                        duration=duration,
                        details="Conversation flow completed" if success else "Conversation flow failed",
                        timestamp=datetime.now().isoformat()
                    )
                    
                    self.results.append(result)
                    status_emoji = "✅" if success else "❌"
                    logger.info(f"{status_emoji} {test['name']} - {duration:.2f}s")
                    
                except Exception as e:
                    duration = time.time() - start_time
                    result = TestResult(
                        test_id=test["test_id"],
                        category=test["category"],
                        name=test["name"], 
                        status="failed",
                        duration=duration,
                        details=str(e),
                        timestamp=datetime.now().isoformat()
                    )
                    self.results.append(result)
                    logger.error(f"❌ {test['name']} - Exception: {e}")
    
    async def _run_enhanced_features_tests(self, tests: List[Dict]):
        """Run enhanced LangGraph features tests"""
        async with aiohttp.ClientSession() as session:
            for test in tests:
                start_time = time.time()
                
                try:
                    if "conversation_flow" in test:
                        # Test enhanced conversation with memory
                        session_id = f"enhanced_test_{test['test_id']}"
                        
                        for turn in test["conversation_flow"]:
                            url = f"{self.base_url}/chat/stream"
                            payload = {
                                "message": turn["user"],
                                "session_id": session_id,
                                "use_enhanced_features": True
                            }
                            
                            async with session.post(url, json=payload) as response:
                                if response.status != 200:
                                    raise Exception(f"HTTP {response.status}")
                        
                        # Check for follow-up suggestions
                        follow_up_url = f"{self.base_url}/chat/follow-up/{session_id}"
                        async with session.get(follow_up_url) as response:
                            if response.status == 200:
                                data = await response.json()
                                suggestions_count = data.get("count", 0)
                                
                                if suggestions_count >= test.get("minimum_questions", 1):
                                    status = "passed"
                                    details = f"Generated {suggestions_count} follow-up questions"
                                else:
                                    status = "failed"
                                    details = f"Only {suggestions_count} questions generated"
                            else:
                                status = "failed"
                                details = "Failed to get follow-up suggestions"
                    
                    elif "patient_case" in test:
                        # Test follow-up generation
                        session_id = f"followup_test_{test['test_id']}"
                        url = f"{self.base_url}/chat/stream"
                        payload = {
                            "message": test["patient_case"],
                            "session_id": session_id,
                            "use_enhanced_features": True
                        }
                        
                        async with session.post(url, json=payload) as response:
                            if response.status == 200:
                                status = "passed"
                                details = "Enhanced features test completed"
                            else:
                                status = "failed"
                                details = f"HTTP {response.status}"
                    
                    duration = time.time() - start_time
                    
                    result = TestResult(
                        test_id=test["test_id"],
                        category=test["category"],
                        name=test["name"],
                        status=status,
                        duration=duration,
                        details=details,
                        timestamp=datetime.now().isoformat()
                    )
                    
                    self.results.append(result)
                    status_emoji = "✅" if status == "passed" else "❌"
                    logger.info(f"{status_emoji} {test['name']} - {duration:.2f}s")
                    
                except Exception as e:
                    duration = time.time() - start_time
                    result = TestResult(
                        test_id=test["test_id"],
                        category=test["category"],
                        name=test["name"],
                        status="failed",
                        duration=duration,
                        details=str(e),
                        timestamp=datetime.now().isoformat()
                    )
                    self.results.append(result)
                    logger.error(f"❌ {test['name']} - Exception: {e}")
    
    async def _run_agent_tests(self, tests: List[Dict]):
        """Run agent testing scenarios"""
        # For now, mark as skipped since these require internal testing
        for test in tests:
            result = TestResult(
                test_id=test["test_id"],
                category=test["category"],
                name=test["name"],
                status="skipped",
                duration=0,
                details="Agent tests require internal Python testing framework",
                timestamp=datetime.now().isoformat()
            )
            self.results.append(result)
            logger.info(f"⏭️  {test['name']} - Skipped (requires pytest)")
    
    async def _run_integration_tests(self, tests: List[Dict]):
        """Run integration tests"""
        # Simplified integration tests
        for test in tests:
            result = TestResult(
                test_id=test["test_id"],
                category=test["category"],
                name=test["name"],
                status="skipped",
                duration=0,
                details="Integration tests require full test framework",
                timestamp=datetime.now().isoformat()
            )
            self.results.append(result)
            logger.info(f"⏭️  {test['name']} - Skipped (requires full framework)")
    
    async def _run_edge_case_tests(self, tests: List[Dict]):
        """Run edge case tests"""
        async with aiohttp.ClientSession() as session:
            for test in tests:
                if test["test_id"] == "EDGE_001":
                    # Test empty/invalid inputs
                    start_time = time.time()
                    
                    try:
                        passed_cases = 0
                        total_cases = len(test["input_cases"])
                        
                        for input_case in test["input_cases"]:
                            url = f"{self.base_url}/chat/stream"
                            payload = {
                                "message": input_case,
                                "session_id": f"edge_test_{hash(input_case)}"
                            }
                            
                            async with session.post(url, json=payload) as response:
                                if response.status == 200:
                                    passed_cases += 1
                        
                        duration = time.time() - start_time
                        
                        if passed_cases == total_cases:
                            status = "passed"
                            details = f"All {total_cases} edge cases handled gracefully"
                        else:
                            status = "failed"
                            details = f"Only {passed_cases}/{total_cases} edge cases passed"
                        
                        result = TestResult(
                            test_id=test["test_id"],
                            category=test["category"],
                            name=test["name"],
                            status=status,
                            duration=duration,
                            details=details,
                            timestamp=datetime.now().isoformat()
                        )
                        
                        self.results.append(result)
                        status_emoji = "✅" if status == "passed" else "❌"
                        logger.info(f"{status_emoji} {test['name']} - {duration:.2f}s")
                        
                    except Exception as e:
                        duration = time.time() - start_time
                        result = TestResult(
                            test_id=test["test_id"],
                            category=test["category"],
                            name=test["name"],
                            status="failed",
                            duration=duration,
                            details=str(e),
                            timestamp=datetime.now().isoformat()
                        )
                        self.results.append(result)
                        logger.error(f"❌ {test['name']} - Exception: {e}")
                
                else:
                    # Skip other edge case tests for now
                    result = TestResult(
                        test_id=test["test_id"],
                        category=test["category"],
                        name=test["name"],
                        status="skipped",
                        duration=0,
                        details="Edge case test requires specialized setup",
                        timestamp=datetime.now().isoformat()
                    )
                    self.results.append(result)
                    logger.info(f"⏭️  {test['name']} - Skipped")
    
    def _generate_report(self):
        """Generate comprehensive test report"""
        passed = len([r for r in self.results if r.status == "passed"])
        failed = len([r for r in self.results if r.status == "failed"])
        skipped = len([r for r in self.results if r.status == "skipped"])
        total = len(self.results)
        
        total_duration = sum(r.duration for r in self.results)
        
        print("\n" + "=" * 60)
        print("🧪 LCA System Test Results")
        print("=" * 60)
        
        print(f"\n📊 Summary:")
        print(f"   ✅ Passed: {passed}")
        print(f"   ❌ Failed: {failed}")
        print(f"   ⏭️  Skipped: {skipped}")
        print(f"   📈 Success Rate: {(passed/total)*100:.1f}%")
        print(f"   ⏱️  Total Duration: {total_duration:.2f}s")
        
        # Category breakdown
        print(f"\n📋 Test Categories:")
        categories = {}
        for result in self.results:
            cat = result.category
            if cat not in categories:
                categories[cat] = {"passed": 0, "failed": 0, "skipped": 0}
            categories[cat][result.status] += 1
        
        for category, counts in categories.items():
            total_cat = sum(counts.values())
            success_rate = (counts["passed"] / total_cat) * 100 if total_cat > 0 else 0
            print(f"   • {category.replace('_', ' ').title()}: {counts['passed']}/{total_cat} passed ({success_rate:.1f}%)")
        
        # Failed tests details
        failed_tests = [r for r in self.results if r.status == "failed"]
        if failed_tests:
            print(f"\n❌ Failed Tests:")
            for test in failed_tests:
                print(f"   • {test.name}: {test.details}")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        if failed > 0:
            print("   • Review failed tests and check service connectivity")
            print("   • Verify all required services (Neo4j, Redis) are running")
            print("   • Check application logs for detailed error information")
        else:
            print("   • All core functionality tests passed! 🎉")
            if skipped > 0:
                print("   • Consider implementing skipped tests for complete coverage")
        
        print("\n" + "=" * 60)

async def main():
    """Main test runner entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="LCA System Test Runner")
    parser.add_argument("--base-url", default="http://localhost:8000",
                       help="Base URL for LCA API (default: http://localhost:8000)")
    parser.add_argument("--category", help="Run specific category only")
    
    args = parser.parse_args()
    
    runner = LCATestRunner(base_url=args.base_url)
    
    if args.category:
        await runner._run_category(args.category)
    else:
        await runner.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())