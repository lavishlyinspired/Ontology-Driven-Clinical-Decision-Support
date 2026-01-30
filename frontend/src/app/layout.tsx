import "./globals.css";
import { Space_Grotesk } from "next/font/google";
import Sidebar from "@/components/Sidebar";

const space = Space_Grotesk({ subsets: ["latin"], variable: "--font-space" });

export const metadata = {
  title: "Lung Cancer Assistant",
  description: "Graph-powered, agentic decision support for lung cancer MDTs."
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={space.variable}>
      <body className="antialiased">
        <div className="app-container">
          {/* Sidebar Navigation */}
          <Sidebar />

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
