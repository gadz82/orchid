import { describe, expect, it } from 'vitest';
import { render } from '@testing-library/react';
import Footer from '@/components/Footer';
import { siteConfig } from '@/site-config';

describe('Footer', () => {
  it('renders on every page: bottom bar carries license, sdd credit and version', () => {
    const { container } = render(<Footer />);
    expect(container.textContent).toContain(`Licensed under ${siteConfig.license}`);
    expect(container.textContent).toContain('Proudly developed using orchid-sdd');
    expect(container.textContent).toContain(`v${siteConfig.version} · Built with Orchid`);
  });

  it('links the orchid-sdd credit to siteConfig.sddUrl', () => {
    const { container } = render(<Footer />);
    const credit = Array.from(container.querySelectorAll('a')).find((a) =>
      a.textContent?.includes('orchid-sdd'),
    );
    expect(credit).toBeDefined();
    expect(credit).toHaveAttribute('href', siteConfig.sddUrl);
  });
});
