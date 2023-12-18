<script lang="ts">
  import { browser } from '$app/environment'
  import Readme from '$root/readme.md'

  const ph_dos_figs = import.meta.glob(`$figs/mace-mp0-phonondb/*-dos-*.svelte`, {
    eager: true,
    import: `default`,
  })
</script>

<main>
  <Readme>
    <img src="/ffonons.svg" alt="FFonons" slot="logo" width="100" />
  </Readme>
</main>

<ul>
  {#if browser}
    {#each Object.values(ph_dos_figs) as ph_dos_fig}
      <svelte:component this={ph_dos_fig} style="max-width: 50cqw;" />
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
  ul {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
    gap: 3em;
    padding: 2em;
    max-width: 95vw;
  }
</style>
