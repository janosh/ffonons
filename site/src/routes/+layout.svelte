<script lang="ts">
  import { goto } from '$app/navigation'
  import { base } from '$app/paths'
  import { page } from '$app/stores'
  import { repository } from '$site/package.json'
  import { CmdPalette } from 'svelte-multiselect'
  import Toc from 'svelte-toc'
  import { CopyButton, GitHubCorner } from 'svelte-zoo'
  import '../app.css'

  $: headingSelector = `main :is(${
    $page.url.pathname === `${base}/api` ? `h1, h2, h3, h4` : `h2`
  }):not(.toc-exclude)`

  const file_routes = Object.keys(import.meta.glob(`./**/+page.{svx,svelte,md}`))
    .filter((key) => !key.includes(`/[`))
    .map((filename) => {
      const parts = filename.split(`/`)
      return `/` + parts.slice(1, -1).join(`/`)
    })

  const actions = file_routes.map((name) => {
    return { label: name, action: () => goto(`${base}${name.toLowerCase()}`) }
  })
</script>

<CmdPalette {actions} placeholder="Go to..." />

<CopyButton global />

<Toc {headingSelector} breakpoint={1250} warnOnEmpty={false} />

<GitHubCorner href={repository} />

<slot />

<footer>
  <a href="{repository}/issues" rel="external">Issues</a>
  <a href="{repository}/discussions" rel="external">Discussions</a>
</footer>

<style>
  footer {
    background: #00061a;
    display: flex;
    flex-wrap: wrap;
    gap: 3ex;
    place-content: center;
    place-items: center;
    margin: 2em 0 0;
    padding: 3vh 3vw;
  }
</style>
