<script lang="ts">
  import { browser } from '$app/environment'
  // import { page } from '$app/stores'
  export let data

    // $: mp_id = input_value.trim().toLowerCase()
    // $: href = `https://legacy.materialsproject.org/materials/${mp_id}`
</script>

<main>
  <!-- <h1><a {href}>MP-{$page.params.slug}</a></h1> -->

  {#if browser}
    <svelte:component this={data.DosFig} style="margin: 4em 0 0;" />
  {/if}
</main>
