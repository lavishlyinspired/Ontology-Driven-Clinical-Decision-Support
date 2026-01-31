'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import {
  Network,
  Home,
  BarChart3,
  Users,
  FileText,
  MessageSquare,
  Activity,
  Database,
  Settings
} from 'lucide-react'

const navItems = [
  { href: "/", label: "Home", icon: Home },
  { href: "/dashboard", label: "Dashboard", icon: BarChart3 },
  { href: "/patients/analyze", label: "Analyze", icon: Users },
  { href: "/guidelines", label: "Guidelines", icon: FileText },
  { href: "/chat", label: "Assistant", icon: MessageSquare },
  { href: "/analytics", label: "Analytics", icon: Activity },
  { href: "/mcp-tools", label: "MCP Tools", icon: Database },
]

interface HeaderProps {
  className?: string
}

export default function Header({ className = '' }: HeaderProps) {
  const pathname = usePathname()

  return (
    <header className={`top-nav ${className}`}>
      <div className="nav-brand">
        <Network className="w-6 h-6 text-white" />
        <div className="brand-text">
          <span className="brand-title">ConsensusCare</span>
          <span className="brand-version">v1.0</span>
        </div>
      </div>

      <nav className="nav-links">
        {navItems.map((item) => {
          const isActive = pathname === item.href ||
            (item.href !== '/' && pathname.startsWith(item.href))

          return (
            <Link
              key={item.href}
              href={item.href}
              className={`nav-item ${isActive ? 'nav-item-active' : ''}`}
            >
              <item.icon className="w-4 h-4" />
              <span>{item.label}</span>
            </Link>
          )
        })}
      </nav>

      <div className="nav-status">
        <div className="status-badge">
          <span className="status-dot"></span>
          <span>Connected</span>
        </div>
      </div>
    </header>
  )
}
