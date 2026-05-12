export const siteConfig = {
  /** Canonical site URL (used for sitemap, social previews, etc.). */
  siteUrl: 'https://gadz82.github.io/orchid/',

  /** Main "orchid" framework repo — shown in Header / Footer. */
  repoUrl: 'https://github.com/gadz82/orchid',

  /**
   * Per-package repositories. Used by package pages to surface a
   * "Source on GitHub" link near the top of the page.
   */
  packageRepos: {
    orchid: 'https://github.com/gadz82/orchid',
    'orchid-api': 'https://github.com/gadz82/orchid-api',
    'orchid-cli': 'https://github.com/gadz82/orchid-cli',
    'orchid-mcp': 'https://github.com/gadz82/orchid-mcp',
    'orchid-frontend': 'https://github.com/gadz82/orchid-frontend',
  } as const,

  /** Maintainer contact channels (Contact page + Footer). */
  contactUrl: '/contact',
  linkedinUrl: 'https://www.linkedin.com/in/francescomarchesini/',

  license: 'MIT',
  version: '0.1.0-poc',
} as const;

export type PackageName = keyof typeof siteConfig.packageRepos;
