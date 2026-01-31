import "./globals.css";
import { Space_Grotesk } from "next/font/google";
import { AppProvider } from "@/contexts/AppContext";
import Header from "@/components/Header";
import { FloatingChatButton } from "@/components/FloatingChatButton";

const space = Space_Grotesk({ subsets: ["latin"], variable: "--font-space" });

export const metadata = {
  title: "Lung Cancer Assistant",
  description: "Graph-powered, agentic decision support for lung cancer MDTs."
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={space.variable}>
      <body className="antialiased">
        <AppProvider>
          <div className="app-layout">
            <Header />
            <div className="app-content">
              {children}
            </div>
            <FloatingChatButton />
          </div>
        </AppProvider>
      </body>
    </html>
  );
}
