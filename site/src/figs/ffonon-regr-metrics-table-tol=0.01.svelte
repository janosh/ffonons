<style type="text/css">
#T_  {
  font-family: sans-serif;
  border-collapse: collapse;
}
#T_ td {
  border: none;
  padding: 4px 6px;
  white-space: nowrap;
}
#T_  th {
  border: none;
  padding: 4px 6px;
  white-space: nowrap;
}
#T_ th.col_heading {
  border: 1px solid;
  border-width: 1px 0;
  text-align: left;
}
#T_ th.row_heading {
  font-weight: normal;
  padding: 3pt;
}
#T__row0_col0, #T__row1_col0, #T__row2_col0, #T__row3_col0 {
  background-color: #08306b;
  color: #f1f1f1;
}
#T__row0_col1 {
  background-color: #99c7e0;
  color: #000000;
}
#T__row0_col2, #T__row1_col1, #T__row2_col2, #T__row3_col2 {
  background-color: #f7fbff;
  color: #000000;
}
#T__row1_col2 {
  background-color: #bad6eb;
  color: #000000;
}
#T__row2_col1 {
  background-color: #1865ac;
  color: #f1f1f1;
}
#T__row3_col1 {
  background-color: #1764ab;
  color: #f1f1f1;
}
table { overflow: scroll; max-width: 100%; display: block; }
table {
    scrollbar-width: none;  /* Firefox */
}
table::-webkit-scrollbar {
    display: none;  /* Safari and Chrome */
}
</style>
<script lang="ts">
      import { sortable } from 'svelte-zoo/actions'
    </script>

    <table use:sortable {...$$props}
     id="T_">
  <thead>
    <tr>
      <th class="index_name level0" >Model</th>
      <th id="T__level0_col0" class="col_heading level0 col0" >MACE-MP</th>
      <th id="T__level0_col1" class="col_heading level0 col1" >CHGNet v0.3.0</th>
      <th id="T__level0_col2" class="col_heading level0 col2" >M3GNet-MS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T__level0_row0" class="row_heading level0 row0" >MAE<sub>ph DOS</sub> (THz) ↓</th>
      <td id="T__row0_col0" class="data row0 col0" >3.87</td>
      <td id="T__row0_col1" class="data row0 col1" >4.83</td>
      <td id="T__row0_col2" class="data row0 col2" >5.44</td>
    </tr>
    <tr>
      <th id="T__level0_row1" class="row_heading level0 row1" >R<sup>2</sup><sub>ph DOS</sub> ↑</th>
      <td id="T__row1_col0" class="data row1 col0" >0.03</td>
      <td id="T__row1_col1" class="data row1 col1" >-0.04</td>
      <td id="T__row1_col2" class="data row1 col2" >-0.02</td>
    </tr>
    <tr>
      <th id="T__level0_row2" class="row_heading level0 row2" >MAE<sub>Ω<sup>max</sup></sub> (THz) ↓</th>
      <td id="T__row2_col0" class="data row2 col0" >1.79</td>
      <td id="T__row2_col1" class="data row2 col1" >2.31</td>
      <td id="T__row2_col2" class="data row2 col2" >4.34</td>
    </tr>
    <tr>
      <th id="T__level0_row3" class="row_heading level0 row3" >R<sup>2</sup><sub>Ω<sup>max</sup></sub> (THz) ↑</th>
      <td id="T__row3_col0" class="data row3 col0" >0.99</td>
      <td id="T__row3_col1" class="data row3 col1" >0.98</td>
      <td id="T__row3_col2" class="data row3 col2" >0.94</td>
    </tr>
  </tbody>
</table>
