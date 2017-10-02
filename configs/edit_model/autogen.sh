#!/bin/bash

options=("attend_pr = 1.0" "lamb_reg = 10.0" "lamb_reg = 25.0" "lamb_reg = 100.0" "edit_dim = 128" "edit_dim = 512" "norm_eps = 0.01" "norm_eps = 0.5" "norm_eps = 1.0" "kill_edit = True")


arraylen=${#options[@]}
#for opt in "${options[@]}"
for (( i=0; i<${arraylen}; i++ ));
do
    echo $i
    echo "include \"edit_baseline.txt\"
editor{
  "${options[$i]}"
}" > configs/edit_model/tmp$i
    ./nlpsub.py -g 1 -n testruns 'python textmorph/edit_model/main.py configs/edit_model/tmp'$i
done
