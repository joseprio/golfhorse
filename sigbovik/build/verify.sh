#!/bin/bash
# usage: ./verify.sh entry.js  — byte count, node output check, gh-verifier-lite
f=${1:-../entry.js}
echo "bytes: $(wc -c < "$f")"
node "$f" | sort | diff - <(sort ../sigbovik.txt) && echo "node output: all 13 hashes, each once"
v=$(ls -d /c/Users/pepri/AppData/Local/Temp/claude/*/*/scratchpad/gh-verifier-lite/verifier.js 2>/dev/null | head -1)
[ -n "$v" ] && node "$v" ../sigbovik.txt "$f"
