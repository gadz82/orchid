import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import CodeBlock, { type CodeTab } from '../CodeBlock';

const TABS: CodeTab[] = [
  { language: 'python', label: 'Python', code: 'print("hello")' },
  { language: 'bash', label: 'Bash', code: 'echo "hello"' },
];

describe('CodeBlock', () => {
  it('renders all tabs', () => {
    render(<CodeBlock tabs={TABS} />);
    expect(screen.getByRole('tab', { name: 'Python' })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: 'Bash' })).toBeInTheDocument();
  });

  it('shows the first tab content by default', () => {
    render(<CodeBlock tabs={TABS} />);
    expect(screen.getByText('print("hello")')).toBeInTheDocument();
  });

  it('switches active tab on click', async () => {
    const user = userEvent.setup();
    render(<CodeBlock tabs={TABS} />);
    await user.click(screen.getByRole('tab', { name: 'Bash' }));
    expect(screen.getByText('echo "hello"')).toBeInTheDocument();
  });

  describe('copy button', () => {
    // jsdom 26 provides a real Clipboard; spy on the actual method rather than
    // replacing the whole navigator.clipboard object (which is non-configurable).
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    let writeTextSpy: any;

    beforeEach(() => {
      writeTextSpy = vi
        .spyOn(navigator.clipboard, 'writeText')
        .mockResolvedValue(undefined);
    });

    afterEach(() => {
      vi.restoreAllMocks();
    });

    it('calls navigator.clipboard.writeText with the active tab code', async () => {
      const user = userEvent.setup();
      render(<CodeBlock tabs={TABS} />);
      await user.click(screen.getByRole('button', { name: /copy code/i }));
      await waitFor(() => {
        expect(writeTextSpy).toHaveBeenCalledWith('print("hello")');
      });
    });

    it('calls writeText with the second tab code after switching', async () => {
      const user = userEvent.setup();
      render(<CodeBlock tabs={TABS} />);
      await user.click(screen.getByRole('tab', { name: 'Bash' }));
      await user.click(screen.getByRole('button', { name: /copy code/i }));
      await waitFor(() => {
        expect(writeTextSpy).toHaveBeenCalledWith('echo "hello"');
      });
    });
  });

  it('renders a tab strip even when only one tab is provided', () => {
    const singleTab: CodeTab[] = [{ language: 'python', label: 'Python', code: 'x = 1' }];
    render(<CodeBlock tabs={singleTab} />);
    expect(screen.getByRole('tablist')).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: 'Python' })).toBeInTheDocument();
  });

  it('single tab appears disabled (aria-disabled=true, tabIndex=-1)', () => {
    const singleTab: CodeTab[] = [{ language: 'typescript', code: 'const x = 1;' }];
    render(<CodeBlock tabs={singleTab} />);
    const tab = screen.getByRole('tab', { name: 'typescript' });
    expect(tab).toHaveAttribute('aria-disabled', 'true');
    expect(tab).toHaveAttribute('tabindex', '-1');
  });

  it('single tab does not switch away when clicked', async () => {
    const user = userEvent.setup();
    const singleTab: CodeTab[] = [{ language: 'python', label: 'Python', code: 'x = 1' }];
    render(<CodeBlock tabs={singleTab} />);
    const tab = screen.getByRole('tab', { name: 'Python' });
    await user.click(tab);
    expect(tab).toHaveAttribute('aria-selected', 'true');
  });

  it('renders filename caption when provided', () => {
    render(<CodeBlock tabs={TABS} filename="example.py" />);
    expect(screen.getByText('example.py')).toBeInTheDocument();
  });

  it('renders highlighted HTML when highlightedHtml is provided', () => {
    const tabs: CodeTab[] = [
      {
        language: 'python',
        code: 'x = 1',
        highlightedHtml: '<pre><code><span class="token">x = 1</span></code></pre>',
      },
    ];
    const { container } = render(<CodeBlock tabs={tabs} />);
    expect(container.innerHTML).toContain('<span class="token">x = 1</span>');
  });
});
