#!/bin/bash
# usage: ./batch.sh "<args1>" "<args2>" ...   — runs sbenc sequentially, appends results to results.txt
for a in "$@"; do
  name=$(echo "$a" | tr -d ' -' )
  r=$(./target/release/sbenc $a --out runs/$name 2>runs/$name.log | tail -1)
  echo "$a => $r" | tee -a results.txt
done
