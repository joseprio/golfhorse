// sbenc — beam-search encoder for the golf.horse `sigbovik` entry.
//
// Decoder being targeted (83 bytes + data):
//   for(w=e=i=1;e OP e/`DATA`.charCodeAt(i/2);)w=++i%65?e.toString(16)[IDX]+w:[console.log(w)]
// with OP in {+=,-=}.  Every loop iteration n (the value of i when the
// condition runs, n = 1,2,...) reads char floor(n/2), updates e, and then
// either prepends hex digit IDX of e.toString(16) to the current line or, when
// (n+1)%65==0, prints the line.  The encoder searches for the cheapest (in
// UTF-8 bytes) character string whose digits reproduce the 13 hashes, in any
// order (the first line must end in the initial w, "1").
//
// The update has to be multiplicative: each char is read twice, and with an
// additive update (e+=1/c, e+=i/c) the second digit is an almost deterministic
// function of the first, so nearly no character can match both.
//
// Search: beam over characters (each char = two iterations); states keep the
// exact double e, the set of finished lines and the current line.  Children
// per parent: 126 one-byte chars (0..127 minus ` \ CR, plus U+FFFD which the
// byte 0xFF decodes to), 1920 two-byte, 61439 three-byte.  Candidate classes
// are sampled per cost level so that only as many children are generated as
// the beam can hold (a random subset of a level is what a beam keeps anyway).

use rayon::prelude::*;
use std::io::Write;

const NLINES: usize = 13;
const NONE: u8 = 255;
const NCHARS: usize = 422; // chars actually read with effect (indices 0..421); char 422 is padding
const MASK52: u64 = (1u64 << 52) - 1;

#[derive(Clone, Copy, PartialEq, Debug)]
enum Hash {
    V6,       // e+=e/c  (or e-=e/c with neg)
    DivPlus1, // e=e/c+1
    DivPlusI, // e=e/c+i
    MulMod,   // e=e*MUL%c
    DivMod,   // e=e/MUL%c   (MUL like .9)
    V5F,      // e=~e*MUL+c with (e&15) digits, in the v6 frame
    DivK,     // e=(e+c)/MUL
    V5D,      // e=~e/MUL+c with (e&15) digits (int32-truncated state)
    CdivE,    // e+=c/e
}

#[derive(Clone, Copy)]
struct Cfg {
    neg: bool,
    idx: i32,
    v5: bool,  // Luke's auroch_v5 decoder: e=~e*MUL+c, digit (e&15), forward, w="" (test mode)
    mul: f64,
    hash: Hash,
}

#[inline(always)]
fn upd_h(cfg: &Cfg, e: f64, n: u32, c: f64) -> f64 {
    match cfg.hash {
        Hash::V6 => upd(cfg.neg, e, c),
        Hash::DivPlus1 => e / c + 1.0,
        Hash::DivPlusI => e / c + n as f64,
        Hash::MulMod => (e * cfg.mul) % c,
        Hash::DivMod => (e / cfg.mul) % c,
        Hash::V5F => upd_v5(cfg.mul, e, c),
        Hash::DivK => (e + c) / cfg.mul,
        Hash::V5D => ((!to_int32(e)) as f64) / cfg.mul + c,
        Hash::CdivE => e + c / e,
    }
}

#[inline(always)]
fn dig_h(cfg: &Cfg, v: f64) -> i32 {
    if cfg.hash == Hash::V5F || cfg.hash == Hash::V5D { dig_v5(v) } else { dig(v, cfg.idx) }
}

#[inline(always)]
fn to_int32(v: f64) -> i32 {
    if !v.is_finite() { return 0; }
    let t = v.trunc();
    let m = t.rem_euclid(4294967296.0);
    (m as u32) as i32
}
#[inline(always)]
fn upd_v5(mul: f64, e: f64, c: f64) -> f64 {
    ((!to_int32(e)) as f64) * mul + c
}
#[inline(always)]
fn dig_v5(v: f64) -> i32 {
    if !v.is_finite() { return -1; }
    to_int32(v) & 15
}

#[inline(always)]
fn upd(neg: bool, e: f64, c: f64) -> f64 {
    let q = e / c;
    if neg {
        e - q
    } else {
        e + q
    }
}

/// Character at index `idx` (>= 1) of v.toString(16) as a hex digit value, or
/// -1 if it is not a fraction digit (i.e. '.', past the end, or v not a
/// normal positive double).  V8's DoubleToRadixCString for radix 16 prints
/// the exact binary expansion: all fraction arithmetic is exact and the
/// rounding branch never fires.
#[inline(always)]
fn dig(v: f64, idx: i32) -> i32 {
    if !(v > 0.0) {
        return -1;
    }
    let bits = v.to_bits();
    let ebits = ((bits >> 52) & 0x7ff) as i32;
    if ebits == 0 || ebits == 0x7ff {
        return -1;
    }
    let ex = ebits - 1023;
    let m = (bits & MASK52) | (1u64 << 52);
    let nint = if ex >= 0 { ex / 4 + 1 } else { 1 };
    let mm = idx - nint;
    if mm < 1 {
        return -1; // '.' or an integer digit (e >= 16^idx never happens)
    }
    let tz = m.trailing_zeros() as i32;
    let plast = 52 - ex - tz;
    let nfrac = (plast + 3) >> 2;
    if mm > nfrac {
        return -1;
    }
    let sh = 52 - ex - 4 * mm;
    if sh >= 64 {
        0
    } else if sh >= 0 {
        ((m >> sh) & 15) as i32
    } else {
        ((m << (-sh)) & 15) as i32
    }
}

/// Reference implementation of the full V8 algorithm (used by --selftest and simulate).
fn v8_to_string16(value: f64) -> String {
    let mut value = value;
    let negative = value < 0.0;
    if negative {
        value = -value;
    }
    let mut integer = value.floor();
    let mut fraction = value - integer;
    let next = f64::from_bits(value.to_bits() + 1);
    let mut delta = 0.5 * (next - value);
    delta = delta.max(f64::from_bits(1));
    let chars = b"0123456789abcdef";
    let mut frac_digits: Vec<u8> = Vec::new();
    if fraction >= delta {
        loop {
            fraction *= 16.0;
            delta *= 16.0;
            let digit = fraction as i32;
            frac_digits.push(chars[digit as usize]);
            fraction -= digit as f64;
            if fraction > 0.5 || (fraction == 0.5 && (digit & 1) == 1) {
                if fraction + delta > 1.0 {
                    loop {
                        match frac_digits.pop() {
                            None => {
                                integer += 1.0;
                                break;
                            }
                            Some(c) => {
                                let d = if c > b'9' { c - b'a' + 10 } else { c - b'0' } as usize;
                                if d + 1 < 16 {
                                    frac_digits.push(chars[d + 1]);
                                    break;
                                }
                            }
                        }
                    }
                    break;
                }
            }
            if !(fraction >= delta) {
                break;
            }
        }
    }
    let mut int_digits: Vec<u8> = Vec::new();
    while (integer / 16.0) >= 9007199254740992.0 * 2.0 {
        integer /= 16.0;
        int_digits.push(b'0');
    }
    loop {
        let rem = integer % 16.0;
        int_digits.push(chars[rem as usize]);
        integer = (integer - rem) / 16.0;
        if !(integer > 0.0) {
            break;
        }
    }
    int_digits.reverse();
    let mut s = String::new();
    if negative {
        s.push('-');
    }
    s.push_str(std::str::from_utf8(&int_digits).unwrap());
    if !frac_digits.is_empty() {
        s.push('.');
        s.push_str(std::str::from_utf8(&frac_digits).unwrap());
    }
    s
}

/// 16-byte beam state.  `pc` = parent index (28 bits) | cost offset from the
/// step's base cost (4 bits); `mc` = index of (mask, cur) in the step's combo
/// table (15 bits) | "char was '$'" flag (bit 15).
#[derive(Clone, Copy)]
struct St {
    e: f64,
    pc: u32,
    ch: u16,
    mc: u16,
}
impl St {
    #[inline(always)]
    fn parent(&self) -> u32 { self.pc & 0x0fff_ffff }
    #[inline(always)]
    fn off(&self) -> u32 { self.pc >> 28 }
    #[inline(always)]
    fn combo(&self) -> usize { (self.mc & 0x7fff) as usize }
    #[inline(always)]
    fn dollar(&self) -> bool { self.mc & 0x8000 != 0 }
}

/// Per-step table of valid (mask, cur) pairs.
struct Combos {
    list: Vec<(u16, u8)>,
    index: Vec<u16>, // [(mask << 4) | cur4] -> index (0xffff = invalid)
}
impl Combos {
    fn build(nfinished: u32, cur_none: bool) -> Combos {
        let mut list = Vec::new();
        let mut index = vec![0xffffu16; 1 << 17];
        for mask in 0u16..(1 << 13) {
            if mask.count_ones() != nfinished { continue; }
            if cur_none {
                index[((mask as usize) << 4) | 15] = list.len() as u16;
                list.push((mask, NONE));
            } else {
                for q in 0..NLINES as u8 {
                    if mask & (1 << q) == 0 {
                        index[((mask as usize) << 4) | q as usize] = list.len() as u16;
                        list.push((mask, q));
                    }
                }
            }
        }
        assert!(list.len() < 0x8000);
        Combos { list, index }
    }
    #[inline(always)]
    fn enc(&self, mask: u16, cur: u8) -> u16 {
        let c = if cur == NONE { 15 } else { cur as usize };
        let v = self.index[((mask as usize) << 4) | c];
        debug_assert!(v != 0xffff);
        v
    }
}

struct Rng(u64);
impl Rng {
    #[inline(always)]
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

struct Problem {
    target: [[u8; 64]; NLINES],
    first: u8,
    cfg: Cfg,
}

/// Event n (loop iteration with i==n before the increment): (is_print, pos).
static BACKWARD: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// Shifted alignment: `charCodeAt(++i/2)` with `w=i%65?…` — string index 0 is dead, char k feeds events (2k-1, 2k).
static SHIFT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
fn event_v5(n: u32) -> (bool, usize) {
    let p = n % 65;
    if p == 64 { (true, 0) } else if BACKWARD.load(std::sync::atomic::Ordering::Relaxed) { (false, 63 - p as usize) } else { (false, p as usize) }
}

fn event(n: u32) -> (bool, usize) {
    let m = n + 1;
    if m % 65 == 0 {
        return (true, 0);
    }
    let l = m / 65;
    let p = m % 65;
    if l == 0 {
        (false, (62 - (p - 2)) as usize)
    } else {
        (false, (63 - (p - 1)) as usize)
    }
}

#[derive(Clone, Copy)]
struct Ev {
    print: bool,
    pos: usize,
}

/// Allowed-digit bitmask for a (mask, cur) at a digit event.
#[inline(always)]
fn allowed(pb: &Problem, ev: Ev, mask: u16, cur: u8) -> u16 {
    if cur != NONE {
        1u16 << pb.target[cur as usize][ev.pos]
    } else {
        let mut a = 0u16;
        for q in 0..NLINES {
            if mask & (1 << q) == 0 {
                a |= 1u16 << pb.target[q][ev.pos];
            }
        }
        a
    }
}

/// Outcomes (mask, cur) of a digit event given digit d.
#[inline(always)]
fn outcomes(pb: &Problem, ev: Ev, mask: u16, cur: u8, d: i32, out: &mut Vec<(u16, u8)>) {
    out.clear();
    if ev.print {
        out.push((mask | (1 << cur), NONE));
        return;
    }
    if d < 0 {
        return;
    }
    let d = d as u8;
    if cur != NONE {
        if pb.target[cur as usize][ev.pos] == d {
            out.push((mask, cur));
        }
    } else {
        for q in 0..NLINES {
            if mask & (1 << q) == 0 && pb.target[q][ev.pos] == d {
                out.push((mask, q as u8));
            }
        }
    }
}

/// Branch-light digit extraction for a block; writes -1 for invalid.
#[inline(always)]
fn dig_block(vals: &[f64], idx: i32, out: &mut [i32; 64]) {
    for k in 0..vals.len() {
        let v = vals[k];
        let bits = v.to_bits();
        let ebits = ((bits >> 52) & 0x7ff) as i32;
        let ex = ebits - 1023;
        let m = (bits & MASK52) | (1u64 << 52);
        let nint = if ex >= 0 { ex / 4 + 1 } else { 1 };
        let mm = idx - nint;
        let tz = m.trailing_zeros() as i32;
        let plast = 52 - ex - tz;
        let nfrac = (plast + 3) >> 2;
        let sh = 52 - ex - 4 * mm;
        let ok = v > 0.0 && ebits != 0 && ebits != 0x7ff && mm >= 1 && mm <= nfrac && sh < 64;
        let d = if sh >= 0 { ((m >> (sh & 63)) & 15) as i32 } else { ((m << ((-sh) & 63)) & 15) as i32 };
        out[k] = if ok { d } else { -1 };
    }
}

#[allow(clippy::too_many_arguments)]
fn expand_parent(
    pb: &Problem,
    j: usize,
    ev0: Ev,
    ev1: Ev,
    p: &St,
    pmask: u16,
    pcur: u8,
    parent_idx: u32,
    cands: &[f64],
    codes: &[u32],
    bytes: u32,
    cnext: &Combos,
    fl0: f64,
    fl1: f64,
    n0: u32,
    n1: u32,
    out: &mut Vec<St>,
    o0: &mut Vec<(u16, u8)>,
    o1: &mut Vec<(u16, u8)>,
) {
    let idx = pb.cfg.idx;
    let neg = pb.cfg.neg;
    let e0 = p.e;
    let off = p.off() + bytes;
    if off > 15 {
        return;
    }
    let pcw = (off << 28) | parent_idx;
    let a0: u16 = if ev0.print { 0xffff } else { allowed(pb, ev0, pmask, pcur) };
    let simple = pcur != NONE && !ev0.print && !ev1.print;
    let a1s: u16 = if simple { allowed(pb, ev1, pmask, pcur) } else { 0 };
    let mc_same = if simple { cnext.enc(pmask, pcur) } else { 0 };
    let mut buf = [0f64; 64];
    let mut dg = [0i32; 64];
    let mut base = 0usize;
    while base < cands.len() {
        let n = (cands.len() - base).min(64);
        let blk = &cands[base..base + n];
        {
            match pb.cfg.hash {
                Hash::V6 => {
                    if neg {
                        for k in 0..n { buf[k] = e0 - e0 / blk[k]; }
                    } else {
                        for k in 0..n { buf[k] = e0 + e0 / blk[k]; }
                    }
                }
                Hash::DivPlus1 => { for k in 0..n { buf[k] = e0 / blk[k] + 1.0; } }
                Hash::DivPlusI => { let nf = n0 as f64; for k in 0..n { buf[k] = e0 / blk[k] + nf; } }
                Hash::MulMod => { let t = e0 * pb.cfg.mul; for k in 0..n { buf[k] = t % blk[k]; } }
                Hash::DivMod => { let t = e0 / pb.cfg.mul; for k in 0..n { buf[k] = t % blk[k]; } }
                Hash::V5F => { let t = (!to_int32(e0)) as f64 * pb.cfg.mul; for k in 0..n { buf[k] = t + blk[k]; } }
                Hash::DivK => { let m = pb.cfg.mul; for k in 0..n { buf[k] = (e0 + blk[k]) / m; } }
                Hash::V5D => { let t = (!to_int32(e0)) as f64 / pb.cfg.mul; for k in 0..n { buf[k] = t + blk[k]; } }
                Hash::CdivE => { for k in 0..n { buf[k] = e0 + blk[k] / e0; } }
            }
            if pb.cfg.hash == Hash::V5F || pb.cfg.hash == Hash::V5D {
                for k in 0..n { dg[k] = dig_v5(buf[k]); }
            } else {
                dig_block(&buf[..n], idx, &mut dg);
            }
        }
        for k in 0..n {
            let d0 = dg[k];
            let two = j > 0 || pb.cfg.v5 || SHIFT.load(std::sync::atomic::Ordering::Relaxed);
            if two && !ev0.print && (d0 < 0 || (a0 >> d0) & 1 == 0) {
                continue;
            }
            let e1 = buf[k];
            if e1 < fl0 {
                continue;
            }
            let c = blk[k];
            let code = codes[base + k];
            if p.dollar() && code == 123 {
                continue;
            }
            let dflag = if code == 36 { 0x8000u16 } else { 0 };
            if two {
                let e2 = upd_h(&pb.cfg, e1, n1, c);
                if e2 < fl1 {
                    continue;
                }
                let d1 = dig_h(&pb.cfg, e2);
                if simple {
                    if d1 >= 0 && (a1s >> d1) & 1 != 0 {
                        out.push(St { e: e2, pc: pcw, ch: code as u16, mc: mc_same | dflag });
                    }
                } else {
                    outcomes(pb, ev0, pmask, pcur, d0, o0);
                    for &(m1, c1) in o0.iter() {
                        outcomes(pb, ev1, m1, c1, d1, o1);
                        for &(m2, c2) in o1.iter() {
                            out.push(St { e: e2, pc: pcw, ch: code as u16, mc: cnext.enc(m2, c2) | dflag });
                        }
                    }
                }
            } else {
                outcomes(pb, ev0, pmask, pcur, d0, o0);
                for &(m1, c1) in o0.iter() {
                    out.push(St { e: e1, pc: pcw, ch: code as u16, mc: cnext.enc(m1, c1) | dflag });
                }
            }
        }
        base += 64;
    }
}

fn encode_bytes(chars: &[u32]) -> Vec<u8> {
    let mut out = Vec::new();
    for &c in chars {
        if c == 0xFFFD {
            out.push(0xFF);
        } else if c < 0x80 {
            out.push(c as u8);
        } else if c < 0x800 {
            out.push(0xC0 | (c >> 6) as u8);
            out.push(0x80 | (c & 0x3F) as u8);
        } else {
            out.push(0xE0 | (c >> 12) as u8);
            out.push(0x80 | ((c >> 6) & 0x3F) as u8);
            out.push(0x80 | (c & 0x3F) as u8);
        }
    }
    out
}

fn decoder_text(cfg: &Cfg) -> (String, String) {
    if cfg.v5 {
        return (
            format!("for(w=e=i=``;e=~e*{}+`", cfg.mul),
            "`.charCodeAt(i/2);)w+=++i%65?(e&15).toString(16):`\n`;console.log(w)".to_string(),
        );
    }
    let op = if cfg.neg { "-=" } else { "+=" };
    let pre = match cfg.hash {
        Hash::V6 => format!("for(w=e=i=1;e{}e/`", op),
        Hash::DivPlus1 | Hash::DivPlusI => "for(w=e=i=1;e=e/`".to_string(),
        Hash::MulMod => format!("for(w=e=i=1;e=e*{}%`", cfg.mul),
        Hash::DivMod => format!("for(w=e=i=1;e=e/{}%`", cfg.mul),
        Hash::V5F => format!("for(w=e=i=1;e=~e*{}+`", cfg.mul),
        Hash::V5D => format!("for(w=e=i=1;e=~e/{}+`", cfg.mul),
        Hash::CdivE => "for(w=e=i=1;e+=`".to_string(),
        Hash::DivK => "for(w=e=i=1;e=(e+`".to_string(),
    };
    if cfg.hash == Hash::V5F || cfg.hash == Hash::V5D {
        return (pre, "`.charCodeAt(i/2);)w=++i%65?(e&15).toString(16)+w:[console.log(w)]".to_string());
    }
    let tail = match cfg.hash {
        Hash::DivPlus1 => "+1",
        Hash::DivPlusI => "+i",
        Hash::DivK => "",
        _ => "",
    };
    if cfg.hash == Hash::DivK {
        return (pre, format!("`.charCodeAt(i/2))/{};)w=++i%65?e.toString(16)[{}]+w:[console.log(w)]", cfg.mul, cfg.idx));
    }
    let shift = SHIFT.load(std::sync::atomic::Ordering::Relaxed);
    let (arg, inc) = if shift { ("++i/2", "i") } else { ("i/2", "++i") };
    if cfg.hash == Hash::CdivE {
        return (pre, format!("`.charCodeAt({})/e;)w={}%65?e.toString(16)[{}]+w:[console.log(w)]", arg, inc, cfg.idx));
    }
    if shift {
        return (pre, format!("`.charCodeAt(++i/2){};)w=i%65?e.toString(16)[{}]+w:[console.log(w)]", tail, cfg.idx));
    }
    (
        pre,
        format!("`.charCodeAt(i/2){};)w=++i%65?e.toString(16)[{}]+w:[console.log(w)]", tail, cfg.idx),
    )
}

/// Simulate the decoder exactly (through the reference toString) and return the printed lines.
fn simulate(cfg: &Cfg, chars: &[u32]) -> Vec<String> {
    let mut lines = Vec::new();
    if cfg.v5 {
        let mut w = String::new();
        let mut e = 0.0f64; // ~"" == -1 == ~0
        let mut i: u32 = 0;
        loop {
            let ci = (i / 2) as usize;
            if ci >= chars.len() { break; }
            e = upd_v5(cfg.mul, e, chars[ci] as f64);
            i += 1;
            if i % 65 == 0 { w.push('\n'); } else { w.push(std::char::from_digit((to_int32(e) & 15) as u32, 16).unwrap()); }
        }
        for l in w.split('\n') { if !l.is_empty() { lines.push(l.to_string()); } }
        return lines;
    }
    let mut w = String::from("1");
    let mut e = 1.0f64;
    let mut i: u32 = 1;
    let shift = SHIFT.load(std::sync::atomic::Ordering::Relaxed);
    loop {
        if shift { i += 1; }
        let ci = (i / 2) as usize;
        if ci >= chars.len() {
            break;
        }
        e = upd_h(cfg, e, if shift { i - 1 } else { i }, chars[ci] as f64);
        if !(e != 0.0) {
            break;
        }
        if !shift { i += 1; }
        if i % 65 == 0 {
            lines.push(w.clone());
            w = String::new();
        } else {
            let s = if cfg.hash == Hash::V5F || cfg.hash == Hash::V5D { format!("{:x}", to_int32(e) & 15) } else { v8_to_string16(e) };
            let ch = if cfg.hash == Hash::V5F || cfg.hash == Hash::V5D { s.chars().next() } else { s.as_bytes().get(cfg.idx as usize).map(|&b| b as char) };
            match ch {
                Some(c) => w.insert(0, c),
                None => w.insert_str(0, "undefined"),
            }
        }
    }
    lines
}

fn selftest() {
    let mut rng = Rng(0x9E3779B97F4A7C15);
    let mut n = 0u64;
    for _ in 0..3_000_000 {
        let ex = (rng.next() % 60) as i32 - 30;
        let mant = rng.next() >> 11;
        let tzr = rng.next() % 8;
        let mant = if tzr == 0 { mant & !((1u64 << (rng.next() % 50)) - 1) } else { mant };
        let v = f64::from_bits(((ex + 1023) as u64) << 52 | (mant & MASK52));
        let s = v8_to_string16(v);
        for idx in 1..16 {
            let want = match s.as_bytes().get(idx) {
                None => -1,
                Some(&b) => {
                    if b == b'.' {
                        -1
                    } else {
                        (b as char).to_digit(16).unwrap() as i32
                    }
                }
            };
            let got = dig(v, idx as i32);
            // integer digits are reported as -1 by dig(); skip those
            let nint = s.find('.').unwrap_or(s.len());
            if idx < nint {
                continue;
            }
            if want != got {
                println!("MISMATCH v={:e} s={} idx={} want={} got={}", v, s, idx, want, got);
                std::process::exit(1);
            }
            n += 1;
        }
    }
    println!("selftest ok ({} digit checks)", n);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut beam: usize = 1_000_000;
    let mut seed: u64 = 1;
    let mut neg = false;
    let mut idx = 12i32;
    let mut out = String::from("best.txt");
    let mut margin = 1.3f64;
    let mut list = String::from("../sigbovik.txt");
    let mut do_selftest = false;
    let mut hashstat = false;
    let mut threads = 0usize;
    let mut no_fffd = false;
    let mut v5 = false;
    let mut freefirst = false; // diagnostic: do not fix the first line to the one ending in '1'
    let mut hash = Hash::V6;
    let mut hash_given = false;
    let mut mul = 9.1f64;
    let mut floor_log2 = 0.0f64; // e must stay >= 2^(-floor_log2 * n / 845) at event n (decay corridor)
    let mut a = 1;
    while a < args.len() {
        match args[a].as_str() {
            "--beam" => { beam = args[a + 1].parse().unwrap(); a += 1; }
            "--seed" => { seed = args[a + 1].parse().unwrap(); a += 1; }
            "--neg" => { neg = true; }
            "--idx" => { idx = args[a + 1].parse().unwrap(); a += 1; }
            "--out" => { out = args[a + 1].clone(); a += 1; }
            "--margin" => { margin = args[a + 1].parse().unwrap(); a += 1; }
            "--list" => { list = args[a + 1].clone(); a += 1; }
            "--threads" => { threads = args[a + 1].parse().unwrap(); a += 1; }
            "--no-fffd" => { no_fffd = true; }
            "--v5" => { v5 = true; }
            "--freefirst" => { freefirst = true; }
            "--backward" => { BACKWARD.store(true, std::sync::atomic::Ordering::Relaxed); }
            "--shift" => { SHIFT.store(true, std::sync::atomic::Ordering::Relaxed); }
            "--hash" => {
                hash_given = true;
                hash = match args[a + 1].as_str() { "v6" => Hash::V6, "div1" => Hash::DivPlus1, "divi" => Hash::DivPlusI, "mulmod" => Hash::MulMod, "divmod" => Hash::DivMod, "v5f" => Hash::V5F, "divk" => Hash::DivK, "v5d" => Hash::V5D, "cde" => Hash::CdivE, _ => panic!("hash") };
                a += 1;
            }
            "--mul" => { mul = args[a + 1].parse().unwrap(); a += 1; }
            "--floor" => { floor_log2 = args[a + 1].parse().unwrap(); a += 1; }
            "--selftest" => { do_selftest = true; }
            "--hashstat" => { hashstat = true; }
            _ => panic!("unknown arg {}", args[a]),
        }
        a += 1;
    }
    if do_selftest {
        selftest();
        return;
    }
    if hashstat {
        // hash quality: for random states, how many distinct (digit1,digit2) pairs do the
        // 126 one-byte / 1920 two-byte candidates produce?  P(some candidate hits a random
        // target pair) = distinct/256.
        if v5 && !hash_given { hash = Hash::V5F; }
        let cfg = Cfg { neg, idx, v5, mul, hash };
        let mut rng = Rng(12345);
        let mut c1: Vec<f64> = Vec::new();
        for c in 0u32..128 { if c != 13 && c != 92 && c != 96 { c1.push(c as f64); } }
        c1.push(65533.0);
        let c2: Vec<f64> = (128u32..0x800).map(|c| c as f64).collect();
        let mut sum1 = 0.0; let mut sum2 = 0.0; let mut cnt = 0;
        for _trial in 0..400 {
            let mut e = if hash == Hash::V5F { 0.0 } else { 1.0 };
            let mut n = 1u32;
            for _ in 0..60 {
                // random walk with 1-byte chars that keep the state valid
                let mut tries = 0;
                loop {
                    let c = c1[(rng.next() % c1.len() as u64) as usize];
                    let e1 = upd_h(&cfg, e, n, c);
                    let e2 = upd_h(&cfg, e1, n + 1, c);
                    if dig_h(&cfg, e1) >= 0 && dig_h(&cfg, e2) >= 0 { e = e2; n += 2; break; }
                    tries += 1; if tries > 1000 { break; }
                }
            }
            for (cands, sum) in [(&c1, &mut sum1), (&c2, &mut sum2)] {
                let mut seen = [false; 256];
                for &c in cands.iter() {
                    let e1 = upd_h(&cfg, e, n, c);
                    let d1 = dig_h(&cfg, e1);
                    if d1 < 0 { continue; }
                    let e2 = upd_h(&cfg, e1, n + 1, c);
                    let d2 = dig_h(&cfg, e2);
                    if d2 < 0 { continue; }
                    seen[(d1 * 16 + d2) as usize] = true;
                }
                *sum += seen.iter().filter(|&&b| b).count() as f64;
            }
            cnt += 1;
        }
        println!("hash {:?} idx {}: distinct pairs per state: 1-byte {:.1}/126 (P(hit)={:.3}, random would be {:.3}), 2-byte {:.1}/256 (P(hit)={:.3}, random {:.3})",
            hash, idx, sum1 / cnt as f64, sum1 / cnt as f64 / 256.0, 1.0 - (255.0f64/256.0).powi(126), sum2 / cnt as f64, sum2 / cnt as f64 / 256.0, 1.0 - (255.0f64/256.0).powi(1920));
        return;
    }
    if threads > 0 {
        rayon::ThreadPoolBuilder::new().num_threads(threads).build_global().unwrap();
    }
    if v5 && !hash_given { hash = Hash::V5F; }
    let cfg = Cfg { neg, idx, v5, mul, hash };
    let text = std::fs::read_to_string(&list).expect("list");
    let mut target = [[0u8; 64]; NLINES];
    let mut first = NONE;
    for (q, line) in text.lines().filter(|l| !l.is_empty()).enumerate() {
        assert_eq!(line.len(), 64);
        for (k, ch) in line.bytes().enumerate() {
            target[q][k] = (ch as char).to_digit(16).unwrap() as u8;
        }
        if target[q][63] == 1 {
            first = q as u8;
        }
    }
    assert!(first != NONE);
    assert_eq!(std::mem::size_of::<St>(), 16);
    assert!(beam < (1usize << 28));
    let pb = Problem { target, first, cfg };

    // candidate classes (code lists and f64 lists)
    let mut codes: [Vec<u32>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for c in 0u32..128 {
        if c == 13 || c == 92 || c == 96 { continue; }
        codes[0].push(c);
    }
    if !no_fffd { codes[0].push(0xFFFD); }
    for c in 128u32..0x800 { codes[1].push(c); }
    for c in 0x800u32..0x10000 {
        if (0xD800..=0xDFFF).contains(&c) || c == 0xFFFD { continue; }
        codes[2].push(c);
    }
    // duplicate the lists so a contiguous window can wrap around without modulo
    let mut codes2: [Vec<u32>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    let mut cands2: [Vec<f64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for k in 0..3 {
        codes2[k] = codes[k].iter().chain(codes[k].iter()).cloned().collect();
        cands2[k] = codes2[k].iter().map(|&c| c as f64).collect();
    }
    let nclass = [codes[0].len(), codes[1].len(), codes[2].len()];
    eprintln!("cfg: hash={:?} neg={} idx={} mul={} v5={} beam={} seed={} classes={:?}", cfg.hash, cfg.neg, cfg.idx, cfg.mul, cfg.v5, beam, seed, nclass);

    // combo table for the initial state (0 finished, cur set)
    let mut ccur = Combos::build(0, v5 || freefirst);
    let mut cur: Vec<St> = if v5 || freefirst {
        vec![St { e: if hash == Hash::V5F || hash == Hash::V5D { 0.0 } else { 1.0 }, pc: 0, ch: 0, mc: ccur.enc(0, NONE) }]
    } else {
        vec![St { e: 1.0, pc: 0, ch: 0, mc: ccur.enc(0, pb.first) }]
    };
    let mut base_cost: usize = 0; // absolute cost of offset 0 in `cur`
    let mut hist: Vec<Vec<(u32, u16)>> = Vec::new();
    let t0 = std::time::Instant::now();

    for j in 0..NCHARS {
        let shift = SHIFT.load(std::sync::atomic::Ordering::Relaxed);
        let (n0, n1) = if v5 { (2 * j as u32, 2 * j as u32 + 1) } else if shift { (2 * j as u32 + 1, 2 * j as u32 + 2) } else if j == 0 { (1u32, 1u32) } else { (2 * j as u32, 2 * j as u32 + 1) };
        let (p0, pos0) = if v5 { event_v5(n0) } else { event(n0) };
        let (p1, pos1) = if v5 { event_v5(n1) } else { event(n1) };
        let ev0 = Ev { print: p0, pos: pos0 };
        let ev1 = Ev { print: p1, pos: pos1 };
        // combos of the children: finished lines = prints so far (through event n1), cur none iff n1 is a print
        let nfin = if v5 { (n1 + 1) / 65 } else { (n1 + 1) / 65 };
        let cnext = Combos::build(nfin, p1);
        let fl0 = if floor_log2 > 0.0 { (2.0f64).powf(-floor_log2 * n0 as f64 / 845.0) } else { f64::NEG_INFINITY };
        let fl1 = if floor_log2 > 0.0 { (2.0f64).powf(-floor_log2 * n1 as f64 / 845.0) } else { f64::NEG_INFINITY };

        let tphase = std::time::Instant::now();
        let omin = cur.iter().map(|s| s.off()).min().unwrap() as usize;
        let omax = cur.iter().map(|s| s.off()).max().unwrap() as usize;
        let cmin = base_cost + omin;
        let cmax = base_cost + omax;
        let mut ncost = vec![0usize; cmax + 1];
        for s in &cur { ncost[base_cost + s.off() as usize] += 1; }
        // per-candidate acceptance probability for this step, averaged over a sample of parents:
        // a print event accepts everything, a known-line digit 1/16, a line start (remaining lines)/16
        let qev = |ev: Ev, mask: u16, cur: u8| -> (f64, u16, u8) {
            if ev.print {
                (1.0, mask | (1 << cur), NONE)
            } else if cur != NONE {
                (1.0 / 16.0, mask, cur)
            } else {
                let rem = NLINES as u32 - (mask as u32).count_ones();
                (rem as f64 / 16.0, mask, 0)
            }
        };
        let stride = (cur.len() / 4096).max(1);
        let mut qsum = 0.0;
        let mut qn = 0usize;
        for s in cur.iter().step_by(stride) {
            let (m, c) = ccur.list[s.combo()];
            let (q0, m1, c1) = qev(ev0, m, c);
            let q = if j == 0 && !v5 && !SHIFT.load(std::sync::atomic::Ordering::Relaxed) { q0 } else { q0 * qev(ev1, m1, c1).0 };
            qsum += q;
            qn += 1;
        }
        let pstep = qsum / qn as f64;
        let pk = [nclass[0] as f64 * pstep, nclass[1] as f64 * pstep, nclass[2] as f64 * pstep];
        let want = beam as f64 * margin;
        let mut cum = 0.0f64;
        let mut rate = vec![[0.0f32; 3]; cmax - cmin + 1];
        for lvl in (cmin + 1)..=(cmax + 3) {
            let mut exp = 0.0;
            for k in 0..3 {
                let pl = lvl as i64 - (k as i64 + 1);
                if pl >= cmin as i64 && pl <= cmax as i64 {
                    exp += ncost[pl as usize] as f64 * pk[k];
                }
            }
            let r = if cum >= want { 0.0 } else if exp <= want - cum { 1.0 } else { (want - cum) / exp };
            cum += exp.min((want - cum).max(0.0));
            for k in 0..3 {
                let pl = lvl as i64 - (k as i64 + 1);
                if pl >= cmin as i64 && pl <= cmax as i64 {
                    rate[(pl - cmin as i64) as usize][k] = r as f32;
                }
            }
        }
        let chunk = ((cur.len() + 511) / 512).max(256);
        let ccur_ref = &ccur;
        let cnext_ref = &cnext;
        let children: Vec<Vec<St>> = cur
            .par_chunks(chunk)
            .enumerate()
            .map(|(ci, ps)| {
                let mut out: Vec<St> = Vec::new();
                let mut o0 = Vec::new();
                let mut o1 = Vec::new();
                let mut rng = Rng(seed.wrapping_mul(0x9E3779B97F4A7C15) ^ ((j as u64) << 32) ^ (ci as u64 + 1).wrapping_mul(0xD1B54A32D192ED03));
                for _ in 0..4 { rng.next(); }
                for (pi, p) in ps.iter().enumerate() {
                    let parent_idx = (ci * chunk + pi) as u32;
                    let (pmask, pcur) = ccur_ref.list[p.combo()];
                    let rates = rate[base_cost + p.off() as usize - cmin];
                    for k in 0..3 {
                        let r = rates[k];
                        if r <= 0.0 { continue; }
                        let n = nclass[k];
                        let (start, take) = if r >= 1.0 {
                            (0usize, n)
                        } else {
                            (((rng.next() % n as u64) as usize), ((n as f32 * r) as usize).max(1))
                        };
                        expand_parent(&pb, j, ev0, ev1, p, pmask, pcur, parent_idx, &cands2[k][start..start + take], &codes2[k][start..start + take], (k + 1) as u32, cnext_ref, fl0, fl1, n0, n1, &mut out, &mut o0, &mut o1);
                    }
                }
                out
            })
            .collect();
        let t_exp = tphase.elapsed().as_secs_f64();
        let total: usize = children.iter().map(|v| v.len()).sum();
        if total == 0 {
            eprintln!("dead end at char {}", j);
            std::process::exit(2);
        }
        // select the cheapest `beam` children (offsets are relative to base_cost)
        let mut hist_c = vec![0usize; 20];
        for v in &children { for s in v { hist_c[s.off() as usize] += 1; } }
        let mut acc = 0usize;
        let mut thr = 19usize;
        let mut room = 0usize;
        for c in 0..hist_c.len() {
            if acc + hist_c[c] > beam { thr = c; room = beam - acc; break; }
            acc += hist_c[c];
        }
        let newmin = hist_c.iter().position(|&n| n > 0).unwrap() as u32;
        let mut kept: Vec<St> = Vec::with_capacity(beam.min(total));
        'outer: for v in &children {
            for s in v {
                let c = s.off() as usize;
                if c < thr { kept.push(*s); }
                else if c == thr {
                    if room > 0 { room -= 1; kept.push(*s); }
                    else if kept.len() >= beam { break 'outer; }
                }
            }
        }
        drop(children);
        // re-base offsets so that the new minimum is offset 0
        for s in kept.iter_mut() {
            let o = s.off() - newmin;
            s.pc = (s.pc & 0x0fff_ffff) | (o << 28);
        }
        let t_sel = tphase.elapsed().as_secs_f64();
        kept.par_sort_unstable_by(|a, b| (a.e.to_bits(), a.combo(), a.off()).cmp(&(b.e.to_bits(), b.combo(), b.off())));
        kept.dedup_by(|a, b| a.e.to_bits() == b.e.to_bits() && a.combo() == b.combo());
        let t_sort = tphase.elapsed().as_secs_f64();
        hist.push(kept.iter().map(|s| (s.parent(), s.ch)).collect());
        let mut lvl = hist.len() - 1;
        while lvl >= 1 {
            let plen = hist[lvl - 1].len();
            let mut used = vec![false; plen];
            for &(p, _) in &hist[lvl] { used[p as usize] = true; }
            let nused = used.iter().filter(|&&u| u).count();
            if nused == plen { break; }
            let mut remap = vec![u32::MAX; plen];
            let mut newv = Vec::with_capacity(nused);
            for (i, &u) in used.iter().enumerate() {
                if u { remap[i] = newv.len() as u32; newv.push(hist[lvl - 1][i]); }
            }
            for ent in hist[lvl].iter_mut() { ent.0 = remap[ent.0 as usize]; }
            hist[lvl - 1] = newv;
            lvl -= 1;
        }
        let t_casc = tphase.elapsed().as_secs_f64();
        base_cost += newmin as usize;
        let kmin = base_cost;
        let kmax = base_cost + kept.iter().map(|s| s.off()).max().unwrap() as usize;
        let nmin = kept.iter().filter(|s| s.off() == 0).count();
        let histmem: usize = hist.iter().map(|v| v.len()).sum();
        if j % 10 == 0 || j >= NCHARS - 8 {
            eprintln!("char {:3} gen {:10} kept {:10} cost {}..{} (n@min {}) hist {} t={:.0}s [exp {:.2} sel {:.2} sort {:.2} casc {:.2}]", j, total, kept.len(), kmin, kmax, nmin, histmem, t0.elapsed().as_secs_f64(), t_exp, t_sel-t_exp, t_sort-t_sel, t_casc-t_sort);
        }
        cur = kept;
        ccur = cnext;
    }
    let mut best: Option<(u32, usize)> = None;
    let mut fin = vec![0usize; 8];
    for (i, s) in cur.iter().enumerate() {
        let (m, c) = ccur.list[s.combo()];
        let full_mask = if c == NONE { m } else { m | (1 << c) };
        if full_mask.count_ones() as usize != NLINES { continue; }
        let d = s.off() as usize;
        if d < 8 { fin[d] += 1; }
        if best.map_or(true, |(o, _)| s.off() < o) { best = Some((s.off(), i)); }
    }
    let (boff, bi) = best.expect("no complete state");
    let bcost = base_cost + boff as usize;
    eprintln!("complete states by cost offset from min: {:?}", fin);
    let mut chars = vec![0u32; NCHARS];
    let mut idxp = bi;
    for j in (0..NCHARS).rev() {
        let (p, ch) = hist[j][idxp];
        chars[j] = ch as u32;
        idxp = p as usize;
    }
    let mut full = chars.clone();
    if SHIFT.load(std::sync::atomic::Ordering::Relaxed) { full.insert(0, b'a' as u32); } else if !v5 { full.push(b'a' as u32); }
    let data = encode_bytes(&full);
    let (pre, post) = decoder_text(&cfg);
    let total_bytes = pre.len() + data.len() + post.len();
    eprintln!("best data cost {} (+{} padding) -> program {} bytes", bcost, if v5 { 0 } else { 1 }, total_bytes);
    let lines = simulate(&cfg, &full);
    let mut ok = lines.len() == NLINES;
    let mut seen = vec![false; NLINES];
    for l in &lines {
        match text.lines().position(|t| t == l) {
            Some(q) if !seen[q] => seen[q] = true,
            _ => ok = false,
        }
    }
    eprintln!("simulation {}", if ok { "OK" } else { "MISMATCH" });
    let mut f = std::fs::File::create(&out).unwrap();
    writeln!(f, "{}", pre).unwrap();
    writeln!(f, "{}", post).unwrap();
    writeln!(f, "{}", full.iter().map(|c| c.to_string()).collect::<Vec<_>>().join(",")).unwrap();
    let mut prog = Vec::new();
    prog.extend_from_slice(pre.as_bytes());
    prog.extend_from_slice(&data);
    prog.extend_from_slice(post.as_bytes());
    std::fs::write(format!("{}.js", out), &prog).unwrap();
    println!("{} {}", bcost, total_bytes);
}
