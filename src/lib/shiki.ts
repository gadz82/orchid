import { createHighlighter, type Highlighter } from 'shiki';
import orchidTheme from './shiki-orchid-theme.json';

let highlighter: Highlighter | null = null;

const SUPPORTED_LANGS = ['python', 'bash', 'yaml', 'typescript', 'tsx', 'json', 'http', 'text'] as const;

async function getHighlighter(): Promise<Highlighter> {
  if (!highlighter) {
    highlighter = await createHighlighter({
      themes: [orchidTheme as Parameters<typeof createHighlighter>[0]['themes'][0]],
      langs: [...SUPPORTED_LANGS],
    });
  }
  return highlighter;
}

export async function highlight(code: string, lang: string): Promise<string> {
  const hl = await getHighlighter();
  const safeLang = (SUPPORTED_LANGS as readonly string[]).includes(lang) ? lang : 'text';
  return hl.codeToHtml(code, { lang: safeLang, theme: 'orchid-dark' });
}
