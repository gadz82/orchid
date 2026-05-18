import { highlight } from '@/lib/shiki';
import CodeBlock, { type CodeTab } from './CodeBlock';

type Props = {
  yaml: string;
  yamlLabel?: string;
  md: string;
  mdLabel?: string;
  filename?: string;
};

export default async function CodePair({
  yaml,
  yamlLabel = 'YAML',
  md,
  mdLabel = 'MD',
  filename,
}: Props) {
  const tabs: CodeTab[] = [];

  try {
    const yamlHtml = await highlight(yaml, 'yaml');
    tabs.push({ language: 'yaml', label: yamlLabel, code: yaml, highlightedHtml: yamlHtml });
  } catch {
    tabs.push({ language: 'yaml', label: yamlLabel, code: yaml });
  }

  try {
    const mdHtml = await highlight(md, 'yaml');
    tabs.push({ language: 'yaml', label: mdLabel, code: md, highlightedHtml: mdHtml });
  } catch {
    tabs.push({ language: 'yaml', label: mdLabel, code: md });
  }

  return <CodeBlock tabs={tabs} filename={filename} />;
}
