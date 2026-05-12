import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import EcosystemSplice from '../EcosystemSplice';

describe('EcosystemSplice', () => {
  it('renders five nodes in full variant', () => {
    render(<EcosystemSplice variant="full" />);
    expect(screen.getAllByTestId('node')).toHaveLength(5);
  });

  it('renders four edges in full variant', () => {
    render(<EcosystemSplice variant="full" />);
    expect(screen.getAllByTestId('edge')).toHaveLength(4);
  });

  it('renders five nodes in compact variant', () => {
    render(<EcosystemSplice variant="compact" />);
    expect(screen.getAllByTestId('node')).toHaveLength(5);
  });

  it('renders four edges in compact variant', () => {
    render(<EcosystemSplice variant="compact" />);
    expect(screen.getAllByTestId('edge')).toHaveLength(4);
  });

  it('defaults to full variant', () => {
    render(<EcosystemSplice />);
    expect(screen.getAllByTestId('node')).toHaveLength(5);
    expect(screen.getAllByTestId('edge')).toHaveLength(4);
  });

  it('shows package names', () => {
    render(<EcosystemSplice />);
    expect(screen.getAllByText('orchid').length).toBeGreaterThan(0);
    expect(screen.getByText('orchid-api')).toBeInTheDocument();
    expect(screen.getByText('orchid-cli')).toBeInTheDocument();
  });
});
