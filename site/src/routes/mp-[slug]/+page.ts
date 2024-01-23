export const prerender = false

export const load = async ({ params }) => {
  const all = import.meta.glob(`$figs/mp-*.svelte`, {
    import: `default`,
  }) as Record<string, () => Promise<ConstructorOfATypedSvelteComponent>>

  // get all figs whose path contains mp-${params.slug}
  const keys = Object.keys(all).filter((path) =>
    path.includes(`mp-${params.slug}`),
  )
  const figs = await Promise.all(keys.map((key) => all[key]()))

  if (!figs) return { status: 404 }

  return { figs }
}
