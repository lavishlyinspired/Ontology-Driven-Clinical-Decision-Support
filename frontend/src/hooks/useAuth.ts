"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";

interface AuthUser {
  user_id: string;
  username: string;
  email: string;
  role: string;
  full_name?: string;
}

export function useAuth() {
  const router = useRouter();
  const [user, setUser] = useState<AuthUser | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    checkAuth();
  }, []);

  const checkAuth = async () => {
    try {
      const token = localStorage.getItem("access_token");
      
      if (!token) {
        setLoading(false);
        return;
      }

      const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
      const response = await fetch(`${apiUrl}/api/v1/auth/me`, {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });

      if (response.ok) {
        const userData = await response.json();
        setUser(userData);
      } else {
        // Token invalid, try refresh
        await refreshToken();
      }
    } catch (error) {
      console.error("Auth check failed:", error);
    } finally {
      setLoading(false);
    }
  };

  const refreshToken = async () => {
    try {
      const refreshToken = localStorage.getItem("refresh_token");
      
      if (!refreshToken) {
        logout();
        return;
      }

      const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
      const response = await fetch(`${apiUrl}/api/v1/auth/refresh`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ refresh_token: refreshToken }),
      });

      if (response.ok) {
        const data = await response.json();
        localStorage.setItem("access_token", data.access_token);
        localStorage.setItem("refresh_token", data.refresh_token);
        await checkAuth();
      } else {
        logout();
      }
    } catch (error) {
      console.error("Token refresh failed:", error);
      logout();
    }
  };

  const logout = () => {
    localStorage.removeItem("access_token");
    localStorage.removeItem("refresh_token");
    setUser(null);
    router.push("/login");
  };

  const getAuthHeaders = () => {
    const token = localStorage.getItem("access_token");
    return token ? { Authorization: `Bearer ${token}` } : {};
  };

  return {
    user,
    loading,
    isAuthenticated: !!user,
    logout,
    refreshToken,
    getAuthHeaders,
  };
}
