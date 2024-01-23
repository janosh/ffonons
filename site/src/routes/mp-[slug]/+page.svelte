<script lang="ts">
  import { browser } from '$app/environment'
  import { page } from '$app/stores'
  export let data

  $: mp_id = $page.params.slug
  $: href = `https://legacy.materialsproject.org/materials/${mp_id}`
</script>

<main>
  <h1><a {href}>MP-{mp_id}</a></h1>

  {#if browser}
    {#each data.figs as fig}
      <svelte:component this={fig} style="margin: 4em 0 0;" />
    {/each}
  {/if}
</main>
