"""
Seed Default Users and Roles
Creates initial admin user and default roles for LCA system
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import uuid

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import auth service after path is set
from backend.src.services.auth_service import auth_service


def seed_default_users():
    """Create default users for LCA system"""
    print("=" * 80)
    print("👥 Seeding Default Users")
    print("=" * 80)
    
    default_users = [
        {
            "username": "admin",
            "email": "admin@lca-system.local",
            "password": "Admin@LCA2026!",  # CHANGE IN PRODUCTION!
            "full_name": "System Administrator",
            "role": "admin"
        },
        {
            "username": "dr_demo",
            "email": "clinician@lca-system.local",
            "password": "Clinician@Demo2026!",  # CHANGE IN PRODUCTION!
            "full_name": "Dr. Demo Clinician",
            "role": "clinician"
        },
        {
            "username": "researcher",
            "email": "researcher@lca-system.local",
            "password": "Researcher@Demo2026!",  # CHANGE IN PRODUCTION!
            "full_name": "Demo Researcher",
            "role": "researcher"
        },
        {
            "username": "viewer",
            "email": "viewer@lca-system.local",
            "password": "Viewer@Demo2026!",  # CHANGE IN PRODUCTION!
            "full_name": "Demo Viewer",
            "role": "viewer"
        }
    ]
    
    created_users = []
    
    for user_data in default_users:
        try:
            # Check if user already exists
            existing_user = auth_service.users_db.get(user_data["username"])
            
            if existing_user:
                print(f"   ⚠ User '{user_data['username']}' already exists - skipping")
                continue
            
            # Create user
            user = auth_service.create_user(
                username=user_data["username"],
                email=user_data["email"],
                password=user_data["password"],
                full_name=user_data["full_name"],
                role=user_data["role"]
            )
            
            created_users.append(user)
            print(f"   ✓ Created user: {user_data['username']} ({user_data['role']})")
            
        except Exception as e:
            print(f"   ❌ Failed to create user '{user_data['username']}': {e}")
    
    print("\n" + "=" * 80)
    print(f"✅ User seeding complete - {len(created_users)} users created")
    print("=" * 80)
    
    if created_users:
        print("\n🔐 Default Credentials (CHANGE IN PRODUCTION!):")
        print("-" * 80)
        for user_data in default_users:
            if any(u["username"] == user_data["username"] for u in created_users):
                print(f"   Username: {user_data['username']:<15} | Password: {user_data['password']}")
        print("-" * 80)
        print("⚠️  WARNING: Change these passwords immediately in production!")
        print("=" * 80)
    
    return created_users


def main():
    """Main entry point"""
    print("\n🌱 Starting User Seed Process...")
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    try:
        users = seed_default_users()
        
        print("\n✅ Seed process completed successfully!")
        print(f"📊 Total users created: {len(users)}")
        
    except Exception as e:
        print(f"\n❌ Seed process failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
