import { describe, it, expect, vi } from 'vitest';
import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import ConfigTable, { anchorId } from '../ConfigTable';

// ── Mock static data ──────────────────────────────────────────────────────────
// vi.mock is hoisted to the top of the file, so data must be inlined in
// the factory functions (no references to outer-scope variables allowed).

vi.mock('@/data/config-schema.json', () => ({
  default: [
    {
      file: 'orchid.yml',
      path: 'llm.model',
      type: 'string',
      required: false,
      default: 'ollama/llama3.2',
      description: 'LiteLLM model string.',
      deprecated: false,
      examples: ['/examples/basketball'],
    },
    {
      file: 'orchid.yml',
      path: 'auth.dev_bypass',
      type: 'boolean',
      required: false,
      default: false,
      description: 'Bypass authentication for development.',
      deprecated: false,
      examples: [],
    },
    {
      file: 'agents.yaml',
      path: 'supervisor.history_max_turns',
      type: 'int',
      required: false,
      default: 20,
      description: 'Maximum conversation turns retained in context.',
      deprecated: false,
      examples: [],
    },
    {
      file: 'agents.yaml',
      path: 'agents[].description',
      type: 'string',
      required: true,
      default: null,
      description: 'Human-readable agent purpose.',
      deprecated: false,
      examples: ['/examples/basketball', '/examples/restaurant'],
    },
  ],
}));

vi.mock('@/data/config-best-practices.json', () => ({
  default: {
    'supervisor.history_max_turns': 'Keep this low for chatbots, higher for research agents.',
  },
}));

// ── Tests ─────────────────────────────────────────────────────────────────────

describe('ConfigTable', () => {
  describe('rendering', () => {
    it('renders only rows matching the file prop', () => {
      render(<ConfigTable file="orchid.yml" />);
      expect(screen.getByText('llm.model')).toBeInTheDocument();
      expect(screen.getByText('auth.dev_bypass')).toBeInTheDocument();
      expect(screen.queryByText('supervisor.history_max_turns')).not.toBeInTheDocument();
    });

    it('renders all agents.yaml rows when file="agents.yaml"', () => {
      render(<ConfigTable file="agents.yaml" />);
      expect(screen.getByText('supervisor.history_max_turns')).toBeInTheDocument();
      expect(screen.getByText('agents[].description')).toBeInTheDocument();
      expect(screen.queryByText('llm.model')).not.toBeInTheDocument();
    });

    it('renders the type column', () => {
      render(<ConfigTable file="orchid.yml" />);
      expect(screen.getByText('string')).toBeInTheDocument();
      expect(screen.getByText('boolean')).toBeInTheDocument();
    });

    it('renders required indicator for required fields', () => {
      render(<ConfigTable file="agents.yaml" />);
      expect(screen.getByText('yes')).toBeInTheDocument();
    });

    it('renders default values', () => {
      render(<ConfigTable file="orchid.yml" />);
      expect(screen.getByText('ollama/llama3.2')).toBeInTheDocument();
    });

    it('renders description text', () => {
      render(<ConfigTable file="orchid.yml" />);
      expect(screen.getByText('LiteLLM model string.')).toBeInTheDocument();
    });
  });

  describe('anchor IDs', () => {
    it('applies stable anchor IDs to orchid.yml rows', () => {
      const { container } = render(<ConfigTable file="orchid.yml" />);
      expect(container.querySelector('#orchid-yml__llm\\.model')).toBeInTheDocument();
      expect(container.querySelector('#orchid-yml__auth\\.dev_bypass')).toBeInTheDocument();
    });

    it('applies stable anchor IDs to agents.yaml rows', () => {
      const { container } = render(<ConfigTable file="agents.yaml" />);
      expect(
        container.querySelector('#agents-yaml__supervisor\\.history_max_turns'),
      ).toBeInTheDocument();
    });

    it('anchorId helper produces the correct ID for orchid.yml', () => {
      expect(anchorId('orchid.yml', 'llm.model')).toBe('orchid-yml__llm.model');
    });

    it('anchorId helper produces the correct ID for agents.yaml', () => {
      expect(anchorId('agents.yaml', 'supervisor.history_max_turns')).toBe(
        'agents-yaml__supervisor.history_max_turns',
      );
    });

    it('anchor IDs are stable across re-renders', () => {
      const { container, rerender } = render(<ConfigTable file="orchid.yml" />);
      const idBefore = container.querySelector('#orchid-yml__llm\\.model')?.id;
      rerender(<ConfigTable file="orchid.yml" />);
      const idAfter = container.querySelector('#orchid-yml__llm\\.model')?.id;
      expect(idBefore).toBe(idAfter);
    });
  });

  describe('filter', () => {
    it('renders the filter input', () => {
      render(<ConfigTable file="orchid.yml" />);
      expect(screen.getByRole('searchbox', { name: /filter/i })).toBeInTheDocument();
    });

    it('narrows rows when typing in the filter', async () => {
      const user = userEvent.setup();
      render(<ConfigTable file="orchid.yml" />);
      const input = screen.getByRole('searchbox', { name: /filter/i });
      await user.type(input, 'model');
      expect(screen.getByText('llm.model')).toBeInTheDocument();
      expect(screen.queryByText('auth.dev_bypass')).not.toBeInTheDocument();
    });

    it('filters by description text', async () => {
      const user = userEvent.setup();
      render(<ConfigTable file="orchid.yml" />);
      const input = screen.getByRole('searchbox', { name: /filter/i });
      await user.type(input, 'bypass');
      expect(screen.getByText('auth.dev_bypass')).toBeInTheDocument();
      expect(screen.queryByText('llm.model')).not.toBeInTheDocument();
    });

    it('shows a results count when filter is active', async () => {
      const user = userEvent.setup();
      render(<ConfigTable file="orchid.yml" />);
      const input = screen.getByRole('searchbox', { name: /filter/i });
      await user.type(input, 'model');
      expect(screen.getByText(/1 result/)).toBeInTheDocument();
    });

    it('shows an empty-state message when no rows match', async () => {
      const user = userEvent.setup();
      render(<ConfigTable file="orchid.yml" />);
      const input = screen.getByRole('searchbox', { name: /filter/i });
      await user.type(input, 'xyznonexistent');
      expect(screen.getByText(/No keys match/)).toBeInTheDocument();
    });

    it('restores all rows after clearing the filter', async () => {
      const user = userEvent.setup();
      render(<ConfigTable file="orchid.yml" />);
      const input = screen.getByRole('searchbox', { name: /filter/i });
      await user.type(input, 'model');
      await user.clear(input);
      expect(screen.getByText('llm.model')).toBeInTheDocument();
      expect(screen.getByText('auth.dev_bypass')).toBeInTheDocument();
    });
  });

  describe('best-practice notes', () => {
    it('renders inline best-practice note for keyed entries', () => {
      render(<ConfigTable file="agents.yaml" />);
      expect(screen.getByText(/Keep this low for chatbots/)).toBeInTheDocument();
    });

    it('does not render best-practice note for entries without one', () => {
      render(<ConfigTable file="orchid.yml" />);
      // llm.model and auth.dev_bypass have no best-practice notes in MOCK_PRACTICES
      expect(screen.queryByText(/Best practice:/)).not.toBeInTheDocument();
    });
  });

  describe('example badges', () => {
    it('renders example badges as links', () => {
      render(<ConfigTable file="orchid.yml" />);
      const link = screen.getByRole('link', { name: 'basketball' });
      expect(link).toHaveAttribute('href', '/examples/basketball');
    });

    it('renders multiple badges for entries with multiple examples', () => {
      render(<ConfigTable file="agents.yaml" />);
      const row = screen.getByText("agents[].description").closest('tr')!;
      const links = within(row).getAllByRole('link');
      expect(links.length).toBeGreaterThanOrEqual(2);
    });

    it('does not render badges for entries with no examples', () => {
      render(<ConfigTable file="orchid.yml" />);
      // auth.dev_bypass has no examples
      const row = screen.getByText('auth.dev_bypass').closest('tr')!;
      const links = within(row).queryAllByRole('link');
      // Only the anchor link on the key itself should be present
      const externalLinks = links.filter(
        (l) => l.getAttribute('href')?.startsWith('/examples') ?? false,
      );
      expect(externalLinks).toHaveLength(0);
    });
  });
});
