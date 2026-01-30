'use client'

import { useState } from 'react'
import Link from 'next/link'
import { ChevronLeft, ChevronRight } from 'lucide-react'

const navItems = [
  { href: "/", label: "Home", icon: "H" },
  { href: "/dashboard", label: "Dashboard", icon: "D" },
  { href: "/patients/analyze", label: "Analyze Patient", icon: "P" },
  { href: "/guidelines", label: "Guidelines", icon: "G" },
  { href: "/analytics", label: "Analytics", icon: "A" },
  { href: "/digital-twin", label: "Digital Twin", icon: "T" },
  { href: "/chat", label: "Chat Assistant", icon: "C" },
  { href: "/mcp-tools", label: "MCP Tools", icon: "M" },
]

export default function Sidebar() {
  const [isCollapsed, setIsCollapsed] = useState(false)

  return (
    <>
      <nav className={`sidebar ${isCollapsed ? 'sidebar-collapsed' : ''}`}>
        <div className="sidebar-header">
          {!isCollapsed && (
            <>
              <h1 className="sidebar-title">🫁 LCA</h1>
              <p className="sidebar-subtitle">v2.0 • Clinical AI</p>
            </>
          )}
          {isCollapsed && (
            <div className="text-center w-full">
              <span className="text-2xl">🫁</span>
            </div>
          )}
        </div>

        {/* Collapse/Expand Button */}
        <button
          onClick={() => setIsCollapsed(!isCollapsed)}
          className="sidebar-toggle-btn"
          title={isCollapsed ? "Expand sidebar" : "Collapse sidebar"}
        >
          {isCollapsed ? (
            <ChevronRight className="w-5 h-5" />
          ) : (
            <ChevronLeft className="w-5 h-5" />
          )}
        </button>

        <ul className="nav-list">
          {navItems.map((item) => (
            <li key={item.href}>
              <Link 
                href={item.href} 
                className={`nav-link group hover:pl-8 transition-all ${isCollapsed ? 'justify-center' : ''}`}
                title={isCollapsed ? item.label : ''}
              >
                <span className="nav-icon bg-blue-50 text-blue-600 group-hover:bg-blue-600 group-hover:text-white transition-colors rounded-lg p-1">
                  {item.icon}
                </span>
                {!isCollapsed && <span className="nav-label">{item.label}</span>}
              </Link>
            </li>
          ))}
        </ul>

        <div className="sidebar-footer">
          <div className={`status-indicator ${isCollapsed ? 'flex-col items-center' : ''}`}>
            <span className="status-dot status-active animate-pulse"></span>
            {!isCollapsed && <span className="font-medium">API Connected</span>}
          </div>
        </div>
      </nav>

      {/* Overlay for mobile */}
      {!isCollapsed && (
        <div 
          className="sidebar-overlay md:hidden" 
          onClick={() => setIsCollapsed(true)}
        />
      )}
    </>
  )
}
