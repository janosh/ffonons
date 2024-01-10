<h1 align="center">
  <slot name="logo">
    <img src="site/static/ffonons.svg" alt="FFonons logo" width="100" /><br />
  </slot>
  FFonons
</h1>

Analysis of foundation model force fields (FF) to predict harmonic phonons.

See [janosh.github.io/ffonons](https://janosh.github.io/ffonons). Very much work in progress! 🚧

Example: Phonon bands and DOS for [mp-2691](https://legacy.materialsproject.org/materials/mp-2691) comparing [CHGNet v0.3.3](https://github.com/CederGroupHub/chgnet/releases/tag/v0.3.3) and [MACE-MP-0](https://arxiv.org/abs/2401.00096) with PBE reference data from Togo's PhononDB (excluding Born non-analytic corrections):

![
  mp-2691-bs-dos-chgnet-v0.3.3-mace-y7uhwpje-pbe
](https://github.com/materialsproject/pymatgen/assets/30958850/d4a3f264-31cb-4033-8a51-979fd031d08e)
