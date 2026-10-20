import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
});

const jetbrainsMono = JetBrains_Mono({
  variable: "--font-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "AstraSearch — Hybrid Search Engine & GraphRAG",
  description:
    "Domain-specific hybrid search engine combining BM25 keyword retrieval, dense vector similarity, Knowledge Graph expansion, and Perplexity-style AI summaries.",
  keywords: ["Hybrid Search", "BM25", "Vector Search", "GraphRAG", "Next.js", "AI Search"],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${inter.variable} ${jetbrainsMono.variable} dark`}>
      <body className="min-h-screen font-sans antialiased selection:bg-indigo-500/30 selection:text-indigo-200">
        {children}
      </body>
    </html>
  );
}
