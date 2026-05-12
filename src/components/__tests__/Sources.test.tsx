import { describe, it, expect } from 'vitest';
import { render } from '@testing-library/react';
import Sources from '../Sources';

describe('Sources', () => {
  it('renders nothing regardless of files passed', () => {
    const { container } = render(<Sources files={['orchid/orchid_ai/core/agent.py']} />);
    expect(container.firstChild).toBeNull();
  });

  it('renders nothing for an empty files array', () => {
    const { container } = render(<Sources files={[]} />);
    expect(container.firstChild).toBeNull();
  });
});
