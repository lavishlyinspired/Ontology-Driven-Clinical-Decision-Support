import "./globals.css";
import { Space_Grotesk } from "next/font/google";
import Link from "next/link";

const space = Space_Grotesk({ subsets: ["latin"], variable: "--font-space" });

export const metadata = {
  title: "Lung Cancer Assistant",
  description: "Graph-powered, agentic decision support for lung cancer MDTs."
};

const navItems = [
  { href: "/", label: "Home", icon: "home" },
  { href: "/dashboard", label: "Dashboard", icon: "dashboard" },
  { href: "/patients/analyze", label: "Analyze Patient", icon: "patient" },
  { href: "/guidelines", label: "Guidelines", icon: "guidelines" },
  { href: "/analytics", label: "Analytics", icon: "analytics" },
  { href: "/digital-twin", label: "Digital Twin", icon: "twin" },
  { href: "/chat", label: "Chat Assistant", icon: "chat" },
  { href: "/mcp-tools", label: "MCP Tools", icon: "tools" },
];

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={space.variable}>
      <body className="antialiased">
        <div className="app-container">
          {/* Sidebar Navigation */}
          <nav className="sidebar">
            <div className="sidebar-header">
              <h1 className="sidebar-title">🫁 LCA</h1>
              <p className="sidebar-subtitle">v2.0 • Clinical AI</p>
            </div>
            <ul className="nav-list">
              {navItems.map((item) => (
                <li key={item.href}>
                  <Link href={item.href} className="nav-link group hover:pl-8 transition-all">
                    <span className="nav-icon bg-blue-50 text-blue-600 group-hover:bg-blue-600 group-hover:text-white transition-colors rounded-lg p-1">{getIcon(item.icon)}</span>
                    <span className="nav-label">{item.label}</span>
                  </Link>
                </li>
              ))}
            </ul>
            <div className="sidebar-footer">
              <div className="status-indicator">
                <span className="status-dot status-active animate-pulse"></span>
                <span className="font-medium">API Connected</span>
              </div>
            </div>
          </nav>

          {/* Main Content */}
          <div className="main-content">
            <div className="gradient-panel">
              <main>{children}</main>
            </div>
          </div>
        </div>
      </body>
    </html>
  );
}

function getIcon(name: string) {
  const icons: Record<string, string> = {
    home: "H",
    dashboard: "D",
    patient: "P",
    guidelines: "G",
    analytics: "A",
    twin: "T",
    chat: "C",
    tools: "M",
  };
  return icons[name] || "?";
}
