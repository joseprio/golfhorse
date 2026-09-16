#!/bin/bash
# usage: ./sweep.sh BEAM IDX SEED_FROM SEED_TO  — appends "seed cost total" lines to sweep_BEAM_IDX.txt, keeps best program
beam=$1; idx=$2; from=$3; to=$4
out=sweep_${beam}_${idx}.txt
for s in $(seq $from $to); do
  r=$(./target/release/sbenc --beam $beam --idx $idx --seed $s --out runs/s_${beam}_${idx}_${s} 2>/dev/null | tail -1)
  echo "$s $r" >> $out
  cost=${r% *}
  best=$(sort -k2 -n $out | head -1 | awk '{print $2}')
  if [ "$cost" != "$best" ]; then rm -f runs/s_${beam}_${idx}_${s} runs/s_${beam}_${idx}_${s}.js; fi
done
