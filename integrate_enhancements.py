"""
Automatic Integration Script for MCP Apps and Clustering
=========================================================

This script automatically integrates all enhancements into conversation_service.py
Run this to complete the implementation.
"""

import re
import shutil
from pathlib import Path

# Paths
backend_path = Path(__file__).parent / "backend" / "src" / "services"
conversation_service = backend_path / "conversation_service.py"
enhancements_file = backend_path / "conversation_service_enhancements.py"

print("🚀 Starting integration...")
print(f"📁 Working with: {conversation_service}")

# 1. Create backup
backup_path = conversation_service.with_suffix('.py.backup')
shutil.copy(conversation_service, backup_path)
print(f"✅ Backup created: {backup_path}")

# 2. Read files
with open(conversation_service, 'r', encoding='utf-8') as f:
    original_content = f.read()

with open(enhancements_file, 'r', encoding='utf-8') as f:
    enhancements_content = f.read()

# 3. Check if already integrated
if '_stream_mcp_app' in original_content and '_stream_clustering_analysis' in original_content:
    print("⚠️  Enhancements already integrated! Skipping...")
    print("✅ Implementation complete - no changes needed")
    exit(0)

# 4. Extract methods from enhancements (skip documentation)
methods_start = enhancements_content.find('async def _stream_mcp_app(')
methods_content = enhancements_content[methods_start:]

# Clean up the methods (remove the standalone function definitions, make them class methods)
# The methods are already properly indented for class methods

# 5. Find where to insert (before the last methods of the class)
# Insert before _format_sse method
insert_point = original_content.find('    def _format_sse(self, data: Dict) -> str:')

if insert_point == -1:
    print("❌ Could not find insertion point!")
    exit(1)

# 6. Insert the new methods
new_content = (
    original_content[:insert_point] +
    "\n    # =========================================================================\n"
    "    # MCP APPS AND CLUSTERING METHODS (Auto-integrated)\n"
    "    # =========================================================================\n\n" +
    methods_content +
    "\n\n    # =========================================================================\n"
    "    # END OF MCP APPS AND CLUSTERING METHODS\n"
    "    # =========================================================================\n\n" +
    original_content[insert_point:]
)

# 7. Write the updated file
with open(conversation_service, 'w', encoding='utf-8') as f:
    f.write(new_content)

print(f"✅ Methods integrated successfully!")
print(f"📊 Original file: {len(original_content)} characters")
print(f"📊 New file: {len(new_content)} characters")
print(f"📊 Added: {len(new_content) - len(original_content)} characters")
print("\n🎉 Integration complete!")
print(f"\n💡 To restore original: cp {backup_path} {conversation_service}")
print("\n🧪 Now test with:")
print("   - Compare treatments")
print("   - Show survival curves")
print("   - Cluster all patients")
