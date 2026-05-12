import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import SearchModal from '@/components/SearchModal';

describe('SearchModal', () => {
  const onClose = vi.fn();

  beforeEach(() => {
    onClose.mockClear();
  });

  it('renders the search input', () => {
    render(<SearchModal onClose={onClose} />);
    expect(screen.getByRole('dialog')).toBeDefined();
    expect(screen.getByPlaceholderText('Search docs…')).toBeDefined();
  });

  it('calls onClose when Escape is pressed', () => {
    render(<SearchModal onClose={onClose} />);
    fireEvent.keyDown(window, { key: 'Escape' });
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('calls onClose when the close button is clicked', () => {
    render(<SearchModal onClose={onClose} />);
    fireEvent.click(screen.getByLabelText('Close search'));
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('calls onClose when the backdrop is clicked', () => {
    render(<SearchModal onClose={onClose} />);
    // The backdrop is the aria-hidden div behind the modal
    const backdrop = document.querySelector('[aria-hidden="true"]') as HTMLElement;
    fireEvent.click(backdrop);
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('shows empty state prompt when no query', () => {
    render(<SearchModal onClose={onClose} />);
    expect(screen.getByText('Type to search the documentation…')).toBeDefined();
  });
});

describe('Header ⌘K integration', () => {
  it('opens SearchModal on ⌘K keydown', async () => {
    const { default: Header } = await import('@/components/Header');
    render(<Header />);
    expect(screen.queryByRole('dialog')).toBeNull();
    fireEvent.keyDown(window, { key: 'k', metaKey: true });
    expect(screen.getByRole('dialog')).toBeDefined();
  });

  it('closes SearchModal on Escape after ⌘K', async () => {
    const { default: Header } = await import('@/components/Header');
    render(<Header />);
    fireEvent.keyDown(window, { key: 'k', metaKey: true });
    expect(screen.getByRole('dialog')).toBeDefined();
    fireEvent.keyDown(window, { key: 'Escape' });
    expect(screen.queryByRole('dialog')).toBeNull();
  });
});
