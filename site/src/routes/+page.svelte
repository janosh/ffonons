<script lang="ts">
  import { browser } from '$app/environment'
  import Readme from '$root/README.md'

  const ph_dos_figs = import.meta.glob(`$figs/mace-mp0-phonondb/*-dos-*.svelte`, {
    eager: true,
    import: `default`,
  })
</script>

<main>
  <Readme />
  {#if browser}
    {#each Object.values(ph_dos_figs) as ph_dos_fig}
      <svelte:component this={ph_dos_fig} />
    {/each}
  {/if}
</main>

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
</style>
