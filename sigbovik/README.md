# golf.horse `sigbovik` entry

`entry.js` is a **549-byte** JavaScript program that prints the 13 SHA-256
digests of the [sigbovik list](http://golf.horse/sigbovik/). The record,
Luke Gustafson's "auroch_v6", is 550 bytes
([write-up](https://www.luke-g.com/sigbovik-golf-horse/),
[part 2](https://www.luke-g.com/packing-hex-numbers-in-javascript-part-2/)),
so this is one byte shorter.

Verified with [gh-verifier-lite](https://github.com/ixchow/gh-verifier-lite)
(`node verifier.js sigbovik.txt entry.js` → `Verified.`); the full
gh-verifier (V8 7.0 built from source) was not run. The program is not valid
UTF-8 on purpose (one byte 0xFF decodes to U+FFFD), exactly like
auroch_v5/v6.

## The problem

13 × 64 hex digits = 3,328 bits of incompressible data, one line starts
with a `0`. There is nothing to model, so the whole game is

1. how many bits a byte of JavaScript source can carry — a template literal
   holds any byte except `` ` ``, `\` and CR, so a 1-byte character is worth
   log2(126) = 6.98 bits (126 because the invalid byte 0xFF decodes to
   U+FFFD, a 126th one-byte character), a 2-byte character 10.9 bits, a
   3-byte one 15.9; mixing them optimally gives ≈ 7.13 bits/byte, i.e. the
   data cannot go below ~462 bytes — and
2. how few bytes turn those characters back into hex digits.

auroch answers (2) with a hash-and-print loop and (1) by *searching* for a
character string whose hash prints the wanted digits, so there is no
unpacking code at all. This entry uses the same 83-byte loop with one
cost-free change, the placement of the `++`:

```js
for(w=e=i=1;e+=e/`DATA`.charCodeAt(i/2);)w=++i%65?e.toString(16)[12]+w:[console.log(w)]   // auroch_v6
for(w=e=i=1;e+=e/`DATA`.charCodeAt(++i/2);)w=i%65?e.toString(16)[12]+w:[console.log(w)]   // entry.js
```

Every loop iteration `n` (the value of `i` when the condition runs) reads
character ⌊n/2⌋, multiplies `e` by `1+1/c`, and then either prepends
character 12 of `e.toString(16)` (a fraction hex digit, roughly bits 41–44
of the mantissa) to the current line or, every 65th iteration, prints the
line (`[console.log(w)]` is `[undefined]`, which stringifies to `""`, so
`w` is reset for free). Each character is read twice and has to produce two
digits, 8 bits, out of its ~8.2 bits of entropy. The loop ends when
`charCodeAt` runs off the string and `e` becomes NaN; one padding character
is needed so that the last print still happens. `w=1` supplies the last
digit of the first line (the one that ends in `1`), and the other 12 lines
may appear in any order, which is worth 12! ≈ 2^29 extra "solutions". 83
bytes of code, 467 bytes of data in auroch_v6 — 466 here.

Moving the `++` shifts which pairs of digits share a character: in auroch
character *k* produces digits 2k and 2k+1 and the string's first character
produces only one digit; here character *k* produces digits 2k−1 and 2k and
the string's *first* character is never read (it is the dead byte instead of
the padding byte at the end). Same byte budget, same code length, but a
completely different set of constraints — and, as it turns out, a different
optimum.

## The encoder (`build/`)

`build/src/main.rs` is a Rust beam search (rayon, 16 threads):

- **Exact semantics.** `toString(16)` on a double is emulated bit-exactly.
  V8's `DoubleToRadixCString` (checked in the 7.0 branch the verifier uses,
  identical in current V8 for radix 16) prints the exact binary expansion
  and its round-to-even branch can never fire for a power-of-two radix, so
  digit 12 is a pure bit-field of the mantissa; `--selftest` checks the fast
  extractor against a transcription of the V8 algorithm on 40 M values, and
  `simulate()` re-runs the final program through the transcription. NUL
  bytes, `${` avoidance and the 0xFF → U+FFFD trick were checked in node.
- **States** are (exact `e`, set of finished lines, current line, cost in
  bytes), packed into 16 bytes; a step is one character = two iterations.
  Candidates: 126 one-byte, 1920 two-byte, 61,439 three-byte characters.
- **Level-aware sampling.** Children of the cheapest parents are enumerated
  exhaustively; for the cost levels that would overflow the beam anyway,
  only a random window of each character class is tried, sized from the
  expected acceptance rate of the step (1/256 for a normal character, much
  higher across a line boundary). A beam keeps a random subset of its top
  level either way, so this loses nothing and makes a step cost ~130
  evaluations per state instead of ~63,000.
- **Genealogy compaction.** Unreferenced ancestors are pruned each step;
  the whole history stays at ~2.3 entries per beam slot because the beam
  coalesces to a few ancestors within ~30 steps. No disk needed.

Results (data bytes include the dead/padding byte; the decoder is 83):

| frame | beam | data | program | time (i7-11800H, 16 threads) |
|---|---|---|---|---|
| auroch alignment | 20 K | 470 | 553 | 7 s |
| auroch alignment | 100 K | 468–470 (60 seeds) | 551–553 | 13 s |
| auroch alignment | 1 M | 467 (10 of 11 seeds) | 550 | 73 s |
| auroch alignment | 16 M | 467 | 550 | 22 min |
| auroch alignment | 64 M, seeds 1 and 2 | 467 | 550 | 92 min |
| auroch alignment | 128 M | 467 | 550 | 3.1 h |
| auroch alignment, digit 10 or 11 | 64 M | 467 | 550 | 92 min |
| **shifted alignment** | **1 M** | **466** | **549** | 73 s |
| shifted alignment, digit 11 / 10 | 1 M | 468 / 469 | 551 / 552 | 73 s |

## What the search is fighting

The beam always holds 4–5 cost levels, each ~100–180× bigger than the one
below, and the cheapest level "dies" every ~10 characters (its handful of
states have no 1-byte continuation, so the minimum jumps by 2). The number
of dies decides the final size, and beyond ~16 M states it no longer
depends on the beam *or the seed*: the 64 M seed-1, 64 M seed-2 and 128 M
runs have identical cheapest-level trajectories from character 320 on (the
same lone states, counts 1, 3, 1, 1, 1). The levels that ever matter are
enumerated exhaustively at that size, so the answer for a given frame is
deterministic: 467 for auroch's alignment (Luke's result exactly — he also
saw no gain from 5 M → 16 M), and the only way to a different answer is a
different frame. The shift of the `++` is such a frame at zero cost; it
gave 466 at a 1 M beam. Beam size buys only logarithmic improvements, as
for any selection-limited branching process (Brunet–Derrida).

Things measured on the way (all at 1 M unless noted):

- Luke's v5 hash `e=~e*9.1+c` with `(e&15)` digits and forward output
  reaches **463** data with a 1 M beam (he needed 16 M × 10 runs). Not
  because its 32-bit state lets paths merge (they barely do), but because
  it is *stratified*: the 126 one-byte characters produce 125 distinct
  (digit, digit) pairs (`--hashstat`), so a state has a 1-byte continuation
  49% of the time instead of 39% for the random-like `e+=e/c`. Its 2-byte
  coverage is poor (160/256), and it likes one output direction: the same
  hash printing lines backwards costs 3 bytes more, and in the v6 frame
  (backwards, fixed first line) it gives 467 like everything else. Its
  decoder is 88–90 bytes, so the best it can do is 551–552.
- Every shorter or bounded-state hash compatible with `toString(16)[k]` is
  worse: additive updates (`e+=1/c`, `e+=i/c`) are broken because the
  second digit of a character is a near-deterministic function of the
  first; `e-=e/c` collapses when `e` underflows the digit and a decay
  corridor costs 4–8 bytes; `e=e/c+1`, `e=e/c+i` are random-like (467);
  `e=(e+c)/k` is degenerate (2^p/k has a small denominator, so consecutive
  characters revisit the same digits); `e=~e/9+c` has a tiny state whose
  exact optimum is 539; `[9]` instead of `[12]` costs 4 data bytes because
  `e` must stay below 16^8.
- Two-digit-per-iteration decoders (`substr(12,2)`, `replace(/./g,…)`,
  literal newlines) are 6–12 bytes longer; BigInt decoders cannot exceed 7
  bits/byte; `encodeURI`/`escape` tricks need bytes that are not valid
  UTF-8.

## Files

| file | purpose |
|---|---|
| `entry.js` | the entry |
| `sigbovik.txt` | the word list |
| `entry.chars.txt` | decoder halves and the code points of the data string |
| `build/src/main.rs` | encoder; `--shift --beam N --idx 12 --seed S --out name` writes `name.js`; `--hash`, `--v5`, `--hashstat`, `--selftest` for the experiments above |
| `build/verify.sh` | byte count, node output check, gh-verifier-lite |
| `build/batch.sh`, `build/sweep.sh` | run many configurations / seeds |

Reproduce: `cd build && cargo build --release && ./target/release/sbenc
--selftest && ./target/release/sbenc --shift --beam 1000000 --out x &&
./verify.sh x.js` (73 s, 16 threads, ~200 MB; gives the 549-byte program).
