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
#T__row0_col0 {
  background-color: #c2d9ee;
  color: #000000;
}
#T__row0_col1, #T__row1_col2, #T__row2_col1, #T__row3_col1, #T__row4_col1, #T__row5_col1, #T__row6_col2 {
  background-color: #f7fbff;
  color: #000000;
}
#T__row0_col2, #T__row1_col1, #T__row2_col2, #T__row3_col2, #T__row4_col2, #T__row5_col2, #T__row6_col1 {
  background-color: #08306b;
  color: #f1f1f1;
}
#T__row1_col0, #T__row6_col0 {
  background-color: #4090c5;
  color: #f1f1f1;
}
#T__row2_col0 {
  background-color: #abd0e6;
  color: #000000;
}
#T__row3_col0 {
  background-color: #91c3de;
  color: #000000;
}
#T__row4_col0 {
  background-color: #8fc2de;
  color: #000000;
}
#T__row5_col0 {
  background-color: #94c4df;
  color: #000000;
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
      <th id="T__level0_col0" class="col_heading level0 col0" >M3GNet-MS</th>
      <th id="T__level0_col1" class="col_heading level0 col1" >CHGNet v0.3.0</th>
      <th id="T__level0_col2" class="col_heading level0 col2" >MACE-MP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T__level0_row0" class="row_heading level0 row0" >Prec. ↑</th>
      <td id="T__row0_col0" class="data row0 col0" >0.61</td>
      <td id="T__row0_col1" class="data row0 col1" >0.53</td>
      <td id="T__row0_col2" class="data row0 col2" >0.83</td>
    </tr>
    <tr>
      <th id="T__level0_row1" class="row_heading level0 row1" >Recall ↑</th>
      <td id="T__row1_col0" class="data row1 col0" >0.93</td>
      <td id="T__row1_col1" class="data row1 col1" >0.99</td>
      <td id="T__row1_col2" class="data row1 col2" >0.81</td>
    </tr>
    <tr>
      <th id="T__level0_row2" class="row_heading level0 row2" >F1 ↑</th>
      <td id="T__row2_col0" class="data row2 col0" >0.74</td>
      <td id="T__row2_col1" class="data row2 col1" >0.69</td>
      <td id="T__row2_col2" class="data row2 col2" >0.82</td>
    </tr>
    <tr>
      <th id="T__level0_row3" class="row_heading level0 row3" >ROC AUC ↑</th>
      <td id="T__row3_col0" class="data row3 col0" >0.67</td>
      <td id="T__row3_col1" class="data row3 col1" >0.56</td>
      <td id="T__row3_col2" class="data row3 col2" >0.83</td>
    </tr>
    <tr>
      <th id="T__level0_row4" class="row_heading level0 row4" >Acc. ↑</th>
      <td id="T__row4_col0" class="data row4 col0" >0.67</td>
      <td id="T__row4_col1" class="data row4 col1" >0.56</td>
      <td id="T__row4_col2" class="data row4 col2" >0.83</td>
    </tr>
    <tr>
      <th id="T__level0_row5" class="row_heading level0 row5" >FPR ↓</th>
      <td id="T__row5_col0" class="data row5 col0" >0.59</td>
      <td id="T__row5_col1" class="data row5 col1" >0.87</td>
      <td id="T__row5_col2" class="data row5 col2" >0.16</td>
    </tr>
    <tr>
      <th id="T__level0_row6" class="row_heading level0 row6" >FNR ↓</th>
      <td id="T__row6_col0" class="data row6 col0" >0.07</td>
      <td id="T__row6_col1" class="data row6 col1" >0.01</td>
      <td id="T__row6_col2" class="data row6 col2" >0.19</td>
    </tr>
  </tbody>
</table>
