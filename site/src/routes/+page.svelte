<script lang="ts">
  import { browser } from '$app/environment'
  import MetricsTable from '$figs/metrics-table.svelte'
  import Readme from '$root/readme.md'

  const figs = import.meta.glob(`$figs/mp-*-bs-dos-*.svelte`, { as: `url`, eager: true })
</script>

<main>
  <Readme>
    <img src="/ffonons.svg" alt="FFonons" slot="logo" width="100" />
    <MetricsTable slot="metrics-table" />
  </Readme>
</main>

<h2>Phonon bands + DOS figures by MP ID</h2>
<p>Hint: Some of these are large files and take a while to load.</p>
<ul>
  {#if browser}
    {#each Object.keys(figs) as key}
      {@const mp_id = `mp-${key.split(`/`).at(-1)?.split(`-`).at(1)}`}
      <li>
        <a href="/{mp_id}">{mp_id}</a>
      </li>
    {/each}
  {/if}
</ul>

<style>
  :global(h1[align='center']) {
    display: flex;
    font-size: clamp(2rem, 2rem + 2vw, 3rem);
    place-items: center;
    place-content: center;
    gap: 6pt;
  }
  :global(a:has(img[alt='Docs'])) {
    display: none;
  }
  h2,
  p {
    text-align: center;
  }
  ul {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(100px, 1fr));
    margin: auto;
    list-style: none;
    gap: 1em;
    padding: 2em;
    max-width: max(80vw, 800px);
  }
</style>
