import { describe, expect, it } from 'vitest';
import { PILLARS } from '@/data/pillars';

/**
 * The home grid promises a complete set of documented, independently linked
 * pillars. Guards: exactly nine cards, every card fully populated, titles
 * unique, and every href points at a site-local route.
 */
describe('PILLARS (home feature grid)', () => {
  it('has exactly nine fully populated cards with unique titles', () => {
    expect(PILLARS).toHaveLength(9);
    const titles = PILLARS.map((p) => p.title);
    expect(new Set(titles).size).toBe(9);
    for (const p of PILLARS) {
      expect(p.title.trim()).not.toBe('');
      expect(p.description.trim()).not.toBe('');
      expect(p.href.startsWith('/')).toBe(true);
    }
  });

  it('covers the persistence capability', () => {
    const persistence = PILLARS.find((p) => p.href === '/concepts/persistence');
    expect(persistence).toBeDefined();
    expect(persistence?.title).toBe('Durable & Resumable');
  });
});
