import { describe, expect, it } from 'vitest';
import { render } from '@testing-library/react';
import Heading, { nodeText, slugify } from '@/components/Heading';

describe('slugify', () => {
  it('lowercases and dashes plain text', () => {
    expect(slugify('The Scope Hierarchy')).toBe('the-scope-hierarchy');
  });

  it('strips punctuation and collapses runs', () => {
    expect(slugify('OrchidRAGScope — typed, *scoped* filters!!')).toBe('orchidragscope-typed-scoped-filters');
  });

  it('keeps code-ish tokens readable', () => {
    expect(slugify('multi_query fan-out with paraphrases')).toBe('multi_query-fan-out-with-paraphrases');
  });
});

describe('nodeText', () => {
  it('flattens nested element trees to text', () => {
    expect(nodeText(['a', ['b', { props: { children: 'c' } }], 'd'])).toBe('abcd');
  });

  it('ignores booleans and nulls', () => {
    expect(nodeText([true, null, 'x'])).toBe('x');
  });
});

describe('Heading', () => {
  it('renders an h2 with a slug id derived from its text', () => {
    const { container } = render(<Heading as="h2">The Scope Hierarchy</Heading>);
    const h2 = container.querySelector('h2');
    expect(h2?.getAttribute('id')).toBe('the-scope-hierarchy');
    expect(h2?.textContent).toContain('The Scope Hierarchy');
  });

  it('omits the id when the text produces no slug', () => {
    const { container } = render(<Heading as="h3">???</Heading>);
    const h3 = container.querySelector('h3');
    expect(h3?.hasAttribute('id')).toBe(false);
  });

  it('exposes an in-page anchor link to the heading', () => {
    const { container } = render(<Heading>Decision points</Heading>);
    expect(container.querySelector('a[href="#decision-points"]')).not.toBeNull();
  });
});
