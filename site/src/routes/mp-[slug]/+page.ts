export const load = async ({ params }) => {
  // return DOS fig matching MP ID
  // site/src/figs/mace-mp0-phonondb/mp-661-dos-pbe-vs-mace-y7uhwpje.svelte

  const ph_dos_figs = import.meta.glob(
    `$figs/mace-mp0-phonondb/*-dos-*.svelte`,
    { import: `default`, eager: true },
  )

  const DosFig =
    await ph_dos_figs[
      `/src/figs/mace-mp0-phonondb/mp-${params.slug}-dos-pbe-vs-mace-y7uhwpje.svelte`
    ]

  return { DosFig }
}
