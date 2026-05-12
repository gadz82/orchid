'use client';

import { usePathname } from 'next/navigation';
import Sidebar from '@/components/Sidebar';

export default function ConditionalSidebar() {
  const pathname = usePathname();
  if (pathname === '/') return null;
  return <Sidebar />;
}
