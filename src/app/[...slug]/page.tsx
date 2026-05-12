import { notFound } from 'next/navigation';
import { readFileSync, existsSync, readdirSync, statSync } from 'fs';
import { join } from 'path';
import matter from 'gray-matter';
import { MDXRemote } from 'next-mdx-remote/rsc';
import remarkGfm from 'remark-gfm';
import type { Metadata } from 'next';
import MdxPre from '@/components/MdxPre';
import Callout from '@/components/Callout';
import ExternalLink from '@/components/ExternalLink';
import MultiLLMBadge from '@/components/MultiLLMBadge';
import EcosystemSplice from '@/components/EcosystemSplice';
import Sources from '@/components/Sources';
import ConfigTable from '@/components/ConfigTable';
import RepoLink from '@/components/RepoLink';
import MdxLink from '@/components/MdxLink';

export const dynamicParams = false;

const CONTENT_DIR = join(process.cwd(), 'src', 'content');

const MDX_COMPONENTS = {
  pre: MdxPre,
  // Internal links in markdown (e.g. [foo](/configuration)) compile to <a> by
  // default, which bypasses Next.js basePath. MdxLink routes same-origin paths
  // through next/link so they get prefixed correctly under GitHub Pages.
  a: MdxLink,
  Callout,
  ExternalLink,
  MultiLLMBadge,
  EcosystemSplice,
  Sources,
  ConfigTable,
  RepoLink,
};

function slugToContentPath(slug: string[]): string | null {
  const direct = join(CONTENT_DIR, ...slug) + '.mdx';
  if (existsSync(direct)) return direct;

  const index = join(CONTENT_DIR, ...slug, 'index.mdx');
  if (existsSync(index)) return index;

  return null;
}

function collectSlugs(dir: string, base: string[] = []): string[][] {
  const results: string[][] = [];
  if (!existsSync(dir)) return results;
  for (const entry of readdirSync(dir)) {
    const full = join(dir, entry);
    if (statSync(full).isDirectory()) {
      results.push(...collectSlugs(full, [...base, entry]));
    } else if (entry.endsWith('.mdx')) {
      const name = entry.replace('.mdx', '');
      if (name === 'index') {
        results.push(base);
      } else {
        results.push([...base, name]);
      }
    }
  }
  return results;
}

export async function generateStaticParams(): Promise<{ slug: string[] }[]> {
  return collectSlugs(CONTENT_DIR).map((slug) => ({ slug }));
}

export async function generateMetadata({ params }: { params: Promise<{ slug: string[] }> }): Promise<Metadata> {
  const { slug } = await params;
  const filePath = slugToContentPath(slug);
  if (!filePath) return {};
  const raw = readFileSync(filePath, 'utf-8');
  const { data } = matter(raw);
  return {
    title: data.title ?? slug.at(-1),
    description: data.description ?? undefined,
  };
}

export default async function ContentPage({ params }: { params: Promise<{ slug: string[] }> }) {
  const { slug } = await params;
  const filePath = slugToContentPath(slug);
  if (!filePath) notFound();

  const raw = readFileSync(filePath, 'utf-8');
  const { data: frontmatter, content } = matter(raw);

  return (
    <article className="prose prose-invert max-w-none [&_a]:text-orchid-accent [&_a:hover]:text-orchid-accent-hover [&_h1]:text-orchid-text [&_h2]:text-orchid-text [&_h3]:text-orchid-text [&_code:not(pre_code)]:bg-orchid-card [&_code:not(pre_code)]:text-orchid-accent-glow [&_code:not(pre_code)]:px-1.5 [&_code:not(pre_code)]:py-0.5 [&_code:not(pre_code)]:rounded [&_code:not(pre_code)]:text-sm [&_pre]:p-0 [&_pre]:bg-transparent">
      <h1>{frontmatter.title}</h1>
      {frontmatter.description && (
        <p className="text-orchid-muted text-lg not-prose mb-6">{frontmatter.description}</p>
      )}
      <MDXRemote source={content} components={MDX_COMPONENTS} options={{ mdxOptions: { remarkPlugins: [remarkGfm] } }} />
    </article>
  );
}
