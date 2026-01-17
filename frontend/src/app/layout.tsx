import "./globals.css";
import { Space_Grotesk } from "next/font/google";

const space = Space_Grotesk({ subsets: ["latin"], variable: "--font-space" });

export const metadata = {
  title: "Lung Cancer Assistant",
  description: "Graph-powered, agentic decision support for lung cancer MDTs."
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={space.variable}>
      <body>
        <div className="gradient-panel">
          <main>{children}</main>
        </div>
      </body>
    </html>
  );
}
