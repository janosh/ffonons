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
#T__row0_col1, #T__row0_col3, #T__row0_col4, #T__row0_col8 {
  background-color: #67001f;
  color: #f1f1f1;
}
#T__row0_col2, #T__row0_col5, #T__row0_col7, #T__row1_col6 {
  background-color: #1a1a1a;
  color: #f1f1f1;
  background-color: #67001f;
  color: #f1f1f1;
}
#T__row0_col6, #T__row1_col2, #T__row1_col5, #T__row1_col7 {
  background-color: #67001f;
  color: #f1f1f1;
  background-color: #1a1a1a;
  color: #f1f1f1;
}
#T__row1_col1, #T__row1_col3, #T__row1_col4, #T__row1_col8 {
  background-color: #1a1a1a;
  color: #f1f1f1;
}
table { overflow: scroll; max-width: 100%; display: block; }
</style>
<script lang="ts">
      import { sortable } from 'svelte-zoo/actions'
    </script>

    <table use:sortable {...$$props}
     id="T_">
  <caption>Metrics for harmonic phonons from ML force fields vs PBE</caption>
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T__level0_col0" class="col_heading level0 col0" >model</th>
      <th id="T__level0_col1" class="col_heading level0 col1" >N</th>
      <th id="T__level0_col2" class="col_heading level0 col2" >MAE<sub>ph DOS</sub></th>
      <th id="T__level0_col3" class="col_heading level0 col3" >R<sup>2</sup><sub>ph DOS</sub></th>
      <th id="T__level0_col4" class="col_heading level0 col4" >Acc. Imag. Modes</th>
      <th id="T__level0_col5" class="col_heading level0 col5" >FPR Imag. Modes</th>
      <th id="T__level0_col6" class="col_heading level0 col6" >FNR Imag. Modes</th>
      <th id="T__level0_col7" class="col_heading level0 col7" >MAE<sub>Ω<sub>max</sub></sub></th>
      <th id="T__level0_col8" class="col_heading level0 col8" >R<sup>2</sup><sub>Ω<sub>max</sub></sub></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T__level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T__row0_col0" class="data row0 col0" >CHGNet v0.3.0</td>
      <td id="T__row0_col1" class="data row0 col1" >137</td>
      <td id="T__row0_col2" class="data row0 col2" >4.69</td>
      <td id="T__row0_col3" class="data row0 col3" >0.02</td>
      <td id="T__row0_col4" class="data row0 col4" >0.55</td>
      <td id="T__row0_col5" class="data row0 col5" >0.89</td>
      <td id="T__row0_col6" class="data row0 col6" >0.00</td>
      <td id="T__row0_col7" class="data row0 col7" >2.39</td>
      <td id="T__row0_col8" class="data row0 col8" >0.98</td>
    </tr>
    <tr>
      <th id="T__level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T__row1_col0" class="data row1 col0" >MACE-MP0</td>
      <td id="T__row1_col1" class="data row1 col1" >243</td>
      <td id="T__row1_col2" class="data row1 col2" >3.89</td>
      <td id="T__row1_col3" class="data row1 col3" >0.04</td>
      <td id="T__row1_col4" class="data row1 col4" >0.81</td>
      <td id="T__row1_col5" class="data row1 col5" >0.16</td>
      <td id="T__row1_col6" class="data row1 col6" >0.22</td>
      <td id="T__row1_col7" class="data row1 col7" >1.78</td>
      <td id="T__row1_col8" class="data row1 col8" >0.99</td>
    </tr>
  </tbody>
</table>
