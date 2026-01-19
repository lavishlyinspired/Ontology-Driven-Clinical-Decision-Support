"""
Authentication API Routes
Handles user authentication, registration, and JWT token management
"""

from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from datetime import datetime

from src.services.auth_service import auth_service
from src.services.audit_service import audit_logger

router = APIRouter(prefix="/auth", tags=["Authentication"])

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


# ==================== Pydantic Models ====================

class UserRegister(BaseModel):
    """User registration request"""
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=8)
    full_name: Optional[str] = None
    role: str = Field(default="clinician", pattern="^(admin|clinician|researcher|viewer)$")

    class Config:
        json_schema_extra = {
            "example": {
                "username": "dr_smith",
                "email": "smith@hospital.com",
                "password": "SecurePassword123!",
                "full_name": "Dr. Jane Smith",
                "role": "clinician"
            }
        }


class UserResponse(BaseModel):
    """User information response"""
    user_id: str
    username: str
    email: str
    full_name: Optional[str]
    role: str
    created_at: datetime
    is_active: bool


class TokenResponse(BaseModel):
    """JWT token response"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class TokenRefresh(BaseModel):
    """Token refresh request"""
    refresh_token: str


class PasswordChange(BaseModel):
    """Password change request"""
    current_password: str
    new_password: str = Field(..., min_length=8)


# ==================== Dependency: Get Current User ====================

async def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    """
    Dependency to extract and validate current user from JWT token
    """
    user_data = auth_service.verify_access_token(token)
    if not user_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user_data


async def get_current_active_user(current_user: dict = Depends(get_current_user)) -> dict:
    """
    Dependency to ensure user is active
    """
    if not current_user.get("is_active", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user account"
        )
    return current_user


# ==================== API Endpoints ====================

@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register_user(user_data: UserRegister):
    """
    Register a new user account
    
    Requires:
    - Unique username and email
    - Strong password (min 8 characters)
    - Valid role
    """
    try:
        # Create user
        user = auth_service.create_user(
            username=user_data.username,
            email=user_data.email,
            password=user_data.password,
            full_name=user_data.full_name,
            role=user_data.role
        )
        
        # Log audit event
        await audit_logger.log_event(
            action="USER_REGISTERED",
            user_id=user["user_id"],
            resource_type="user",
            resource_id=user["user_id"],
            details={"username": user["username"], "role": user["role"]}
        )
        
        return UserResponse(**user)
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.post("/login", response_model=TokenResponse)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Authenticate user and return JWT tokens
    
    Uses OAuth2 password flow (form-encoded username/password)
    """
    # Authenticate user
    user = auth_service.authenticate_user(form_data.username, form_data.password)
    
    if not user:
        # Log failed login attempt
        await audit_logger.log_event(
            action="LOGIN_FAILED",
            user_id=form_data.username,
            resource_type="auth",
            resource_id="login",
            details={"reason": "invalid_credentials"}
        )
        
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Generate tokens
    access_token = auth_service.create_access_token({"sub": user["user_id"], "role": user["role"]})
    refresh_token = auth_service.create_refresh_token({"sub": user["user_id"]})
    
    # Log successful login
    await audit_logger.log_event(
        action="LOGIN_SUCCESS",
        user_id=user["user_id"],
        resource_type="auth",
        resource_id="login",
        details={"username": user["username"]}
    )
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=1800  # 30 minutes
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(token_data: TokenRefresh):
    """
    Refresh access token using refresh token
    """
    # Verify refresh token
    payload = auth_service.verify_refresh_token(token_data.refresh_token)
    
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token"
        )
    
    user_id = payload.get("sub")
    
    # Get user to include role in new access token
    user = auth_service.get_user(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    
    # Generate new tokens
    access_token = auth_service.create_access_token({"sub": user_id, "role": user["role"]})
    new_refresh_token = auth_service.create_refresh_token({"sub": user_id})
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=new_refresh_token,
        expires_in=1800
    )


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: dict = Depends(get_current_active_user)):
    """
    Get current authenticated user information
    """
    user = auth_service.get_user(current_user["sub"])
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    return UserResponse(**user)


@router.post("/change-password")
async def change_password(
    password_data: PasswordChange,
    current_user: dict = Depends(get_current_active_user)
):
    """
    Change user password (requires current password)
    """
    user_id = current_user["sub"]
    
    # Verify current password
    user = auth_service.get_user(user_id)
    if not auth_service.verify_password(password_data.current_password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect"
        )
    
    # Update password
    success = auth_service.update_user_password(user_id, password_data.new_password)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update password"
        )
    
    # Log password change
    await audit_logger.log_event(
        action="PASSWORD_CHANGED",
        user_id=user_id,
        resource_type="user",
        resource_id=user_id,
        details={}
    )
    
    return {"message": "Password changed successfully"}


@router.post("/logout")
async def logout(current_user: dict = Depends(get_current_active_user)):
    """
    Logout user (client should discard tokens)
    
    Note: Since we use stateless JWT, actual token invalidation
    would require a token blacklist (future enhancement)
    """
    # Log logout
    await audit_logger.log_event(
        action="LOGOUT",
        user_id=current_user["sub"],
        resource_type="auth",
        resource_id="logout",
        details={}
    )
    
    return {"message": "Logged out successfully"}


@router.get("/users", response_model=list[UserResponse])
async def list_users(current_user: dict = Depends(get_current_active_user)):
    """
    List all users (admin only)
    """
    # Check admin permission
    if current_user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    users = auth_service.list_users()
    return [UserResponse(**user) for user in users]
