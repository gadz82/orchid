import type { Metadata, Viewport } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';
import Header from '@/components/Header';
import ConditionalSidebar from '@/components/ConditionalSidebar';
import Footer from '@/components/Footer';

const inter = Inter({ subsets: ['latin'], variable: '--font-inter' });

export const metadata: Metadata = {
  title: {
    template: '%s · Orchestrator Index',
    default: 'Orchestrator Index — Multi-Agent AI Framework',
  },
  description: 'Platform-agnostic multi-agent AI framework built on LangGraph.',
};

export const viewport: Viewport = {
  themeColor: '#0D0B11',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={inter.variable}>
      <body className="antialiased bg-orchid-bg text-orchid-text min-h-screen flex flex-col">
        <Header />
        <div className="flex flex-1 max-w-screen-xl mx-auto w-full px-6 gap-8">
          <ConditionalSidebar />
          <main id="main-content" className="flex-1 min-w-0 py-8">
            {children}
          </main>
        </div>
        <Footer />
      </body>
    </html>
  );
}
