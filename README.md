# jordancoblin.github.io

This is the source code for my personal website [jordancoblin.github.io](https://jordancoblin.github.io), built with [Hugo](https://gohugo.io/) and the [Paper theme](https://themes.gohugo.io/themes/hugo-paper/).

## Getting Started

To run the website locally, make sure you have [Hugo](https://gohugo.io/getting-started/installing/) installed.

To run the development server, use the following command:

```bash
hugo server -D
```

This will start a local server at `http://localhost:1313`, where you can preview the website.

## Configuration

The main configuration file is `hugo.toml`. Here are some important settings you can customize:

```toml
title = 'AI Meanderings'
theme = 'paper'
publishDir = '../docs'
```

## Creating a New Post

To create a new blog post, create a new Markdown file in the `content/posts` directory, run the following command:

```bash
hugo new posts/your-post-title.md
```

## Deployment

The website is deployed to GitHub Pages using the `docs` folder as the publishing directory. To deploy the site, run:

```bash
hugo
```

Then, commit and push the changes to the `main` branch of your repository. GitHub Pages will automatically serve the content from the `docs` folder.
