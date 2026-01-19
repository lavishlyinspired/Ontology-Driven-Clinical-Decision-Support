"""
Authentication & Authorization Service

Implements JWT-based authentication with Role-Based Access Control (RBAC)
for secure access to the LCA system.
"""

from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from passlib.context import CryptContext
from jose import JWTError, jwt
from pydantic import BaseModel, EmailStr
from enum import Enum
import secrets


# Configuration
SECRET_KEY = secrets.token_urlsafe(32)  # Generate secure secret key
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7


class UserRole(str, Enum):
    """User roles with hierarchical permissions."""
    ADMIN = "admin"
    CLINICIAN = "clinician"
    VIEWER = "viewer"
    RESEARCHER = "researcher"


class Permission(str, Enum):
    """Granular permissions."""
    # Patient operations
    READ_PATIENT = "read:patient"
    WRITE_PATIENT = "write:patient"
    DELETE_PATIENT = "delete:patient"
    
    # Analysis operations
    RUN_ANALYSIS = "run:analysis"
    VIEW_ANALYSIS = "view:analysis"
    EXPORT_ANALYSIS = "export:analysis"
    
    # System operations
    MANAGE_USERS = "manage:users"
    VIEW_AUDIT_LOGS = "view:audit_logs"
    MANAGE_SETTINGS = "manage:settings"
    
    # Advanced operations
    BATCH_PROCESS = "batch:process"
    OVERRIDE_RECOMMENDATION = "override:recommendation"


# Role to permissions mapping
ROLE_PERMISSIONS: Dict[UserRole, List[Permission]] = {
    UserRole.ADMIN: [
        # All permissions
        Permission.READ_PATIENT,
        Permission.WRITE_PATIENT,
        Permission.DELETE_PATIENT,
        Permission.RUN_ANALYSIS,
        Permission.VIEW_ANALYSIS,
        Permission.EXPORT_ANALYSIS,
        Permission.MANAGE_USERS,
        Permission.VIEW_AUDIT_LOGS,
        Permission.MANAGE_SETTINGS,
        Permission.BATCH_PROCESS,
        Permission.OVERRIDE_RECOMMENDATION,
    ],
    UserRole.CLINICIAN: [
        # Clinical operations
        Permission.READ_PATIENT,
        Permission.WRITE_PATIENT,
        Permission.RUN_ANALYSIS,
        Permission.VIEW_ANALYSIS,
        Permission.EXPORT_ANALYSIS,
        Permission.OVERRIDE_RECOMMENDATION,
    ],
    UserRole.RESEARCHER: [
        # Research operations
        Permission.READ_PATIENT,
        Permission.VIEW_ANALYSIS,
        Permission.EXPORT_ANALYSIS,
        Permission.BATCH_PROCESS,
    ],
    UserRole.VIEWER: [
        # Read-only
        Permission.READ_PATIENT,
        Permission.VIEW_ANALYSIS,
    ],
}


class User(BaseModel):
    """User model."""
    user_id: str
    email: EmailStr
    username: str
    full_name: str
    role: UserRole
    department: Optional[str] = None
    is_active: bool = True
    created_at: datetime
    last_login: Optional[datetime] = None


class UserInDB(User):
    """User model with hashed password."""
    hashed_password: str


class Token(BaseModel):
    """JWT token response."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class TokenData(BaseModel):
    """Data stored in JWT token."""
    user_id: str
    username: str
    role: UserRole
    permissions: List[str]


class PasswordHasher:
    """Secure password hashing using bcrypt."""
    
    def __init__(self):
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    
    def hash_password(self, password: str) -> str:
        """Hash a password."""
        return self.pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against hash."""
        return self.pwd_context.verify(plain_password, hashed_password)


class AuthService:
    """Authentication service handling JWT tokens and user verification."""
    
    def __init__(self, secret_key: str = SECRET_KEY):
        self.secret_key = secret_key
        self.password_hasher = PasswordHasher()
        
        # In-memory user store (replace with database in production)
        self.users_db: Dict[str, UserInDB] = {}
        self.refresh_tokens: Dict[str, str] = {}  # token -> user_id
    
    def create_user(
        self,
        email: str,
        username: str,
        password: str,
        full_name: str,
        role: UserRole,
        department: Optional[str] = None
    ) -> User:
        """Create a new user."""
        user_id = f"user_{secrets.token_urlsafe(16)}"
        
        hashed_password = self.password_hasher.hash_password(password)
        
        user_in_db = UserInDB(
            user_id=user_id,
            email=email,
            username=username,
            full_name=full_name,
            role=role,
            department=department,
            is_active=True,
            created_at=datetime.now(),
            hashed_password=hashed_password
        )
        
        self.users_db[username] = user_in_db
        
        return User(**user_in_db.dict(exclude={'hashed_password'}))
    
    def authenticate_user(self, username: str, password: str) -> Optional[UserInDB]:
        """Authenticate user with username and password."""
        user = self.users_db.get(username)
        if not user:
            return None
        
        if not user.is_active:
            return None
        
        if not self.password_hasher.verify_password(password, user.hashed_password):
            return None
        
        # Update last login
        user.last_login = datetime.now()
        
        return user
    
    def create_access_token(
        self, 
        user: UserInDB,
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create JWT access token."""
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
        permissions = [p.value for p in ROLE_PERMISSIONS[user.role]]
        
        to_encode = {
            "sub": user.user_id,
            "username": user.username,
            "role": user.role.value,
            "permissions": permissions,
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access"
        }
        
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=ALGORITHM)
        return encoded_jwt
    
    def create_refresh_token(self, user: UserInDB) -> str:
        """Create refresh token."""
        expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        
        to_encode = {
            "sub": user.user_id,
            "exp": expire,
            "type": "refresh"
        }
        
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=ALGORITHM)
        
        # Store refresh token
        self.refresh_tokens[encoded_jwt] = user.user_id
        
        return encoded_jwt
    
    def login(self, username: str, password: str) -> Optional[Token]:
        """Login user and return tokens."""
        user = self.authenticate_user(username, password)
        if not user:
            return None
        
        access_token = self.create_access_token(user)
        refresh_token = self.create_refresh_token(user)
        
        return Token(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
        )
    
    def verify_token(self, token: str) -> Optional[TokenData]:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[ALGORITHM])
            
            user_id: str = payload.get("sub")
            username: str = payload.get("username")
            role_str: str = payload.get("role")
            permissions: List[str] = payload.get("permissions", [])
            
            if user_id is None or username is None:
                return None
            
            role = UserRole(role_str)
            
            return TokenData(
                user_id=user_id,
                username=username,
                role=role,
                permissions=permissions
            )
        except JWTError:
            return None
    
    def refresh_access_token(self, refresh_token: str) -> Optional[str]:
        """Refresh access token using refresh token."""
        try:
            payload = jwt.decode(refresh_token, self.secret_key, algorithms=[ALGORITHM])
            
            if payload.get("type") != "refresh":
                return None
            
            user_id: str = payload.get("sub")
            
            # Find user
            user = None
            for u in self.users_db.values():
                if u.user_id == user_id:
                    user = u
                    break
            
            if not user:
                return None
            
            # Create new access token
            access_token = self.create_access_token(user)
            return access_token
            
        except JWTError:
            return None
    
    def logout(self, refresh_token: str):
        """Logout user by invalidating refresh token."""
        if refresh_token in self.refresh_tokens:
            del self.refresh_tokens[refresh_token]
    
    def get_user(self, username: str) -> Optional[User]:
        """Get user by username."""
        user_in_db = self.users_db.get(username)
        if user_in_db:
            return User(**user_in_db.dict(exclude={'hashed_password'}))
        return None
    
    def list_users(self) -> List[User]:
        """List all users."""
        return [
            User(**u.dict(exclude={'hashed_password'}))
            for u in self.users_db.values()
        ]
    
    def update_user_role(self, username: str, new_role: UserRole) -> bool:
        """Update user role."""
        if username in self.users_db:
            self.users_db[username].role = new_role
            return True
        return False
    
    def deactivate_user(self, username: str) -> bool:
        """Deactivate user."""
        if username in self.users_db:
            self.users_db[username].is_active = False
            return True
        return False


class AuthorizationService:
    """Authorization service for permission checking."""
    
    @staticmethod
    def has_permission(token_data: TokenData, required_permission: Permission) -> bool:
        """Check if user has required permission."""
        return required_permission.value in token_data.permissions
    
    @staticmethod
    def has_any_permission(token_data: TokenData, permissions: List[Permission]) -> bool:
        """Check if user has any of the required permissions."""
        return any(p.value in token_data.permissions for p in permissions)
    
    @staticmethod
    def has_all_permissions(token_data: TokenData, permissions: List[Permission]) -> bool:
        """Check if user has all required permissions."""
        return all(p.value in token_data.permissions for p in permissions)
    
    @staticmethod
    def has_role(token_data: TokenData, required_role: UserRole) -> bool:
        """Check if user has required role."""
        return token_data.role == required_role
    
    @staticmethod
    def is_admin(token_data: TokenData) -> bool:
        """Check if user is admin."""
        return token_data.role == UserRole.ADMIN


# Global service instances
auth_service = AuthService()
authorization_service = AuthorizationService()


# Helper function to initialize default users
def initialize_default_users():
    """Create default admin and test users."""
    # Admin user
    auth_service.create_user(
        email="admin@lca.hospital.org",
        username="admin",
        password="Admin@123",  # Change in production!
        full_name="System Administrator",
        role=UserRole.ADMIN,
        department="IT"
    )
    
    # Clinician user
    auth_service.create_user(
        email="dr.smith@lca.hospital.org",
        username="dr_smith",
        password="Clinician@123",
        full_name="Dr. John Smith",
        role=UserRole.CLINICIAN,
        department="Oncology"
    )
    
    # Viewer user
    auth_service.create_user(
        email="viewer@lca.hospital.org",
        username="viewer",
        password="Viewer@123",
        full_name="Clinical Viewer",
        role=UserRole.VIEWER,
        department="Oncology"
    )
    
    print("✅ Default users created:")
    print("   Admin: admin / Admin@123")
    print("   Clinician: dr_smith / Clinician@123")
    print("   Viewer: viewer / Viewer@123")
