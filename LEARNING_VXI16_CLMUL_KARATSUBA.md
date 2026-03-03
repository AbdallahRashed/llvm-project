# Using Legal vXi8 CLMUL to Improve Wider CLMUL Codegen via Karatsuba Decomposition

## Table of Contents
1. [The Task: What Were We Asked to Do?](#1-the-task-what-were-we-asked-to-do)
2. [Background: CLMUL, CLMULH, CLMULR — What Are They?](#2-background-clmul-clmulh-clmulr--what-are-they)
3. [What Does "Legal" Mean in LLVM?](#3-what-does-legal-mean-in-llvm)
4. [The Problem: Why Was Wide CLMUL So Expensive?](#4-the-problem-why-was-wide-clmul-so-expensive)
5. [The ExpandIntRes_CLMUL Pattern (Karatsuba Identity)](#5-the-expandintres_clmul-pattern-karatsuba-identity)
6. [What Are Lo/Hi Halves and Merging?](#6-what-are-lohi-halves-and-merging)
7. [Our Solution: Three Generic Strategies](#7-our-solution-three-generic-strategies)
8. [The Circular Expansion Trap](#8-the-circular-expansion-trap)
9. [The BITREVERSE Bottleneck Fix](#9-the-bitreverse-bottleneck-fix)
10. [Results: Before vs After](#10-results-before-vs-after)
11. [The Stretch Goal: Scalar CLMUL via FPU Transfer](#11-the-stretch-goal-scalar-clmul-via-fpu-transfer)
12. [Key Takeaways](#12-key-takeaways)
13. [Relationship to Previous Bugs](#13-relationship-to-previous-bugs)

---

## 1. The Task: What Were We Asked to Do?

### The Issue

[GitHub Issue #183768](https://github.com/llvm/llvm-project/issues/183768):
**"[AArch64] Can we use legal vXi8 CLMUL to improve vXi16/vXi32/Xi64 CLMUL codegen"**

### The Exact Request (Quoted from the Issue)

> Can we reuse the ExpandIntRes_CLMUL pattern to help aarch64 vXi16/vXi32/Xi64
> CLMUL codegen by using vXi8 clmul support to create lo/hi halves for merging
>
> Preferably this needs to be done generically but it might have to be done in
> AArch64ISelLowering
>
> A stretch goal would investigate whether scalar clmul expansion would benefit
> from transferring to/from the fpu to process as the lowest element of a
> v8i8/v4i16/v2i32/v1i64 type.

The issue points to the `ExpandIntRes_CLMUL` function in `LegalizeIntegerTypes.cpp` (lines 5483-5506), which already implements a Karatsuba decomposition for **scalar** type splitting. The question is: can we apply the same mathematical identity **element-wise on vectors**?

### Breaking Down What Was Asked

| Requirement | What It Means | Where to Do It |
|-------------|---------------|----------------|
| "reuse ExpandIntRes_CLMUL pattern" | Apply Karatsuba decomposition element-wise | `TargetLowering.cpp::expandCLMUL()` |
| "using vXi8 clmul support" | Decompose v*i16/i32/i64 to eventually reach v*i8 PMUL | Generic, benefits AArch64 |
| "create lo/hi halves for merging" | Split wide elements into narrow halves, compute, reassemble | Truncate + Shift + ZExt + OR |
| "done generically" | Put it in TargetLowering (shared across all targets) | `llvm/lib/CodeGen/SelectionDAG/TargetLowering.cpp` |
| "might have to be done in AArch64ISelLowering" | If generic isn't feasible, fallback to target-specific | `llvm/lib/Target/AArch64/AArch64ISelLowering.cpp` |
| "stretch goal: scalar via FPU" | Use FMOV to move scalar to NEON, use PMUL, FMOV back | Not implemented (future work) |

### What We Actually Delivered

| Deliverable | Status |
|-------------|--------|
| Generic Karatsuba in `expandCLMUL()` (Strategy 1) | ✅ Done |
| Generic element-width widening (Strategy 2) | ✅ Done (also helps x86) |
| Generic element-count widening (Strategy 3) | ✅ Done (handles v4i16→v8i16) |
| `canNarrowCLMULToLegal()` helper to prevent cycles | ✅ Done |
| Custom BITREVERSE for v4i16/v8i16 on AArch64 | ✅ Done (needed for CLMULH path) |
| Updated test expectations (AArch64 + x86) | ✅ Done |
| Scalar CLMUL via FPU transfer | ❌ Not done (stretch goal, separate patch) |

### The Observation

AArch64 (ARM 64-bit) has hardware support for carryless multiplication on **byte-sized elements only** — the PMUL instruction works on `v8i8` (8 bytes) and `v16i8` (16 bytes). There is **no hardware PMUL** for 16-bit, 32-bit, or 64-bit elements.

When you ask the compiler for CLMUL on `v8i16` (8 × 16-bit), `v4i32` (4 × 32-bit), etc., LLVM previously had no choice but to expand it **bit-by-bit**: a loop of shifts, ANDs, and XORs. For a 16-bit CLMUL, that's 16 iterations. For 32-bit, that's 32 iterations. The result: **~70 instructions for v8i16, ~130 instructions for v4i32**.

### The Insight

But wait — we know how to decompose a 16-bit CLMUL using Karatsuba into **three** 8-bit multiplications: one XLo×YLo (giving both low and high parts), plus two cross terms (XLo×YHi and XHi×YLo). And 8-bit CLMULs are **free** on AArch64 (PMUL instruction)! So instead of 70 instructions of bit-manipulation, we should be able to do ~18 instructions of actual vector operations.

### Files Modified

```
llvm/lib/CodeGen/SelectionDAG/TargetLowering.cpp   — Core: 3 strategies + canNarrowCLMULToLegal()
llvm/lib/Target/AArch64/AArch64ISelLowering.cpp     — Custom BITREVERSE for v4i16/v8i16
llvm/test/CodeGen/AArch64/clmul-fixed.ll            — Updated expected output
llvm/test/CodeGen/X86/clmul-vector.ll               — Updated expected output
llvm/test/CodeGen/X86/clmul-vector-256.ll           — Updated expected output
```

---

## 2. Background: CLMUL, CLMULH, CLMULR — What Are They?

### Carryless Multiplication (CLMUL)

Normal multiplication adds partial products. Carryless multiplication **XORs** them instead — no carries propagate.

```
Regular multiply:           Carryless multiply (CLMUL):
    1101  (13)                  1101
  ×   11  (3)                ×   11
  ------                     ------
    1101                       1101
 + 1101                      ^ 1101       ← XOR, not ADD
  ------                     ------
  100111  (39)                10001
```

This is used in:
- **AES-GCM** encryption (HTTPS, TLS)
- **CRC32** checksums (networking, storage)
- **Polynomial arithmetic** over GF(2)

### The Three Variants

Think of CLMUL as producing a **double-width result**. For two 8-bit inputs, the full carryless product is 15 bits wide (fits in 16 bits):

```
CLMUL(A, B) where A and B are each 8 bits:

Full result = [15-bit product]

Split into two halves:
  ┌─────────────────────────────────────────────┐
  │ Bit 14  ...  Bit 8 │ Bit 7  ...  Bit 0     │
  │   HIGH half (CLMULH)│   LOW half (CLMUL)    │
  └─────────────────────────────────────────────┘

CLMUL(A, B)  = Lower 8 bits of the full product
CLMULH(A, B) = Upper 8 bits of the full product (shifted right by 1 bit*)
CLMULR(A, B) = Reversed version: bitreverse(CLMUL(bitreverse(A), bitreverse(B)))
```

*Note: CLMULH shifts by `BW` (the element bit-width), and CLMULR shifts by `BW - 1`.

### Why CLMULH Matters for This Task

CLMULH gives us the **overflow** — the bits that "spill above" the element width. When we decompose a 16-bit CLMUL into 8-bit operations, we need **both** the low result (CLMUL) and the overflow (CLMULH) of the 8-bit sub-multiplications. This is the core of the Karatsuba identity.

### Why CLMULR Matters

CLMULR is expanded via the **bitreverse path**:
```
CLMULR(X, Y) = bitreverse(CLMUL(bitreverse(X), bitreverse(Y)))
```
This creates a CLMUL operation that benefits from our Karatsuba strategies. So improving CLMUL automatically improves CLMULR.

---

## 3. What Does "Legal" Mean in LLVM?

### The Legality System

LLVM's backend categorizes every (operation, type) pair as one of:

| Status | Meaning | Example |
|--------|---------|---------|
| **Legal** | Hardware instruction exists | CLMUL on v8i8 (PMUL) |
| **Custom** | Target has a special lowering | CLMUL on v4i32 for x86 (PCLMULQDQ) |
| **Expand** | Must be synthesized from simpler ops | CLMUL on v8i16 on AArch64 |
| **Promote** | Widen to a larger type and try again | i16 ADD → promote to i32 ADD |

### What's Legal for CLMUL?

**AArch64 (NEON):**
```cpp
setOperationAction(ISD::CLMUL, {MVT::v8i8, MVT::v16i8}, Legal);
// That's it! Only byte-sized elements.
// v8i16, v4i32, v2i64, etc. → Expand (bit-by-bit)
```

**x86 (with PCLMUL):**
```cpp
setOperationAction(ISD::CLMUL, {MVT::v4i32, MVT::v2i64}, Custom);
// With AVX2: also MVT::v8i32, MVT::v4i64
// With AVX512+VPCLMULQDQ: also MVT::v16i32, MVT::v8i64
```

**Key observation**: x86 has legal CLMUL on 32-bit and 64-bit elements, but AArch64 only has 8-bit. We need to bridge that gap — use 8-bit operations to implement wider ones.

### What Does "Legal Type" Mean?

A type is **legal** if the target hardware has registers that can hold it. For AArch64 NEON:

| Type | Legal? | Why |
|------|--------|-----|
| v8i8 | ✅ | Fits in 64-bit D register |
| v16i8 | ✅ | Fits in 128-bit Q register |
| v8i16 | ✅ | 128-bit, fits in Q register |
| v4i16 | ✅ | 64-bit, fits in D register |
| v4i32 | ✅ | 128-bit, fits in Q register |
| v2i32 | ✅ | 64-bit, fits in D register |
| v4i8 | ❌ | 32-bit, no 32-bit NEON register |
| v16i16 | ❌ | 256-bit, exceeds NEON max (128-bit) |

This matters: when we halve v4i16 (4×16-bit) to v4i8 (4×8-bit), v4i8 is **not a legal type** — so Karatsuba can't be applied directly. We need Strategy 3 (widen element count) to get to a legal type first.

---

## 4. The Problem: Why Was Wide CLMUL So Expensive?

### The Bit-by-Bit Expansion

Before our change, `expandCLMUL()` had only one strategy for CLMUL when the type wasn't legal: expand it bit-by-bit.

For `CLMUL(v8i16)` (8 × 16-bit elements), this means:

```cpp
// Pseudocode of the old bit-by-bit expansion:
Result = 0
for I = 0 to 15:           // 16 iterations for 16-bit elements!
    Mask = (1 << I)
    YBit = Y & Mask         // Isolate bit I of Y
    if YBit != 0:
        Result ^= (X << I) // XOR shifted X into result
return Result
```

Each iteration produces:
- AND (mask)
- SETCC (compare to zero)
- SHL (shift X)
- SELECT (conditional)
- XOR (accumulate)

That's **~5 operations × 16 iterations = ~80 operations per element lane**. Across 8 vector lanes, many merge into SIMD instructions, but we still end up with **~70 actual instructions** for v8i16.

### The Core Waste

The tragedy: AArch64 **can** do CLMUL — just on 8-bit elements. A 16-bit carryless multiply is mathematically decomposable into four 8-bit carryless multiplies. But the compiler didn't know this trick.

---

## 5. The ExpandIntRes_CLMUL Pattern (Karatsuba Identity)

### What Is Karatsuba?

The Karatsuba algorithm is a **divide-and-conquer** approach to multiplication. It was discovered in 1960 by Anatoly Karatsuba and reduces multiplication of two n-digit numbers from n² operations to about n^1.585 operations.

### How It Applies to CLMUL

Karatsuba works even better for carryless multiplication because there are **no carries** — the sub-results combine purely with XOR.

The identity for carryless multiplication:

```
Given: X and Y are N-bit values
Split: X = (XHi : XLo), Y = (YHi : YLo)  — each half is N/2 bits

CLMUL(X, Y) = (Hi << N/2) | Lo

where:
  Lo = CLMUL(XLo, YLo)                              — low × low
  Hi = CLMULH(XLo, YLo) ^ CLMUL(XLo, YHi) ^ CLMUL(XHi, YLo)  — cross terms
```

### Visual: Decomposing a 16-bit CLMUL into 8-bit Operations

```
X (16-bit) = [ XHi (8-bit) | XLo (8-bit) ]
Y (16-bit) = [ YHi (8-bit) | YLo (8-bit) ]

CLMUL(X, Y):

    XHi    XLo
     ×      ×
    YHi    YLo

    ┌──────────────────────────────────────┐
    │  CLMUL(XLo, YLo) = [LoH : Lo]       │  ← produces 15-bit result
    │  split into Lo (lower 8) and LoH     │
    │  (upper 8, which is CLMULH(XLo,YLo)) │
    ├──────────────────────────────────────┤
    │  CLMUL(XLo, YHi) = Cross1           │  ← XOR into Hi
    │  CLMUL(XHi, YLo) = Cross2           │  ← XOR into Hi
    ├──────────────────────────────────────┤
    │  Hi = LoH ^ Cross1 ^ Cross2         │
    │  Result = (Hi << 8) | Lo             │
    └──────────────────────────────────────┘
```

### The Existing Code in ExpandIntRes_CLMUL

This pattern already existed in LLVM for **scalar type splitting** — when a scalar integer is too wide (e.g., i128 on a 64-bit target), LLVM splits it into two halves:

```cpp
// In LegalizeIntegerTypes.cpp:
void DAGTypeLegalizer::ExpandIntRes_CLMUL(SDNode *N, SDValue &Lo, SDValue &Hi) {
  // Split X and Y into halves
  GetExpandedInteger(N->getOperand(0), LL, LH);  // X → (XLo, XHi)
  GetExpandedInteger(N->getOperand(1), RL, RH);  // Y → (YLo, YHi)

  Lo = DAG.getNode(ISD::CLMUL, DL, HalfVT, LL, RL);          // Lo = CLMUL(XLo, YLo)
  SDValue LoH = DAG.getNode(ISD::CLMULH, DL, HalfVT, LL, RL); // overflow
  SDValue Cross1 = DAG.getNode(ISD::CLMUL, DL, HalfVT, LL, RH);
  SDValue Cross2 = DAG.getNode(ISD::CLMUL, DL, HalfVT, LH, RL);
  Hi = LoH ^ Cross1 ^ Cross2;                                  // XOR cross terms
}
```

**Our innovation**: Reuse this same identity **element-wise on vectors** in `expandCLMUL()`, not just for scalar type splitting.

---

## 6. What Are Lo/Hi Halves and Merging?

### The Concept

When we split a wide value into halves, we get:
- **Lo (low half)**: The least significant bits
- **Hi (high half)**: The most significant bits

```
A 16-bit value: 0xABCD

  Hi half (8 bits)    Lo half (8 bits)
  ┌──────────────┐   ┌──────────────┐
  │     0xAB     │   │     0xCD     │
  └──────────────┘   └──────────────┘
  bits [15:8]         bits [7:0]
```

### Splitting (Extracting Halves)

For vectors, we split element-wise using TRUNCATE and SRL:

```cpp
// Extract halves from each 16-bit element:
SDValue XLo = DAG.getNode(ISD::TRUNCATE, DL, v8i8, X);        // Keep lower 8 bits
SDValue XHi = DAG.getNode(ISD::TRUNCATE, DL, v8i8,
                 DAG.getNode(ISD::SRL, DL, v8i16, X, 8));     // Shift right 8, truncate
```

Visual for one element:
```
X element = 0b 1010_1100 0011_0101  (16 bits)
                ^^^^^^^^  ^^^^^^^^
                   XHi       XLo

XLo = TRUNCATE(X) = 0b 0011_0101    (lower 8 bits)
XHi = TRUNCATE(X >> 8) = 0b 1010_1100  (upper 8 bits)
```

### Merging (Reassembling from Halves)

After computing the Karatsuba sub-results, we reassemble the full-width result:

```cpp
// Merge: Result = ZExt(Lo) | (ZExt(Hi) << HalfBW)
SDValue LoExt = DAG.getNode(ISD::ZERO_EXTEND, DL, v8i16, Lo);   // 8→16 bits, zero-padded
SDValue HiExt = DAG.getNode(ISD::ZERO_EXTEND, DL, v8i16, Hi);
SDValue HiShifted = DAG.getNode(ISD::SHL, DL, v8i16, HiExt, 8); // Shift left 8
return DAG.getNode(ISD::OR, DL, v8i16, LoExt, HiShifted);       // Combine
```

Visual:
```
Lo = 0b 0110_1001 (8 bits)     Hi = 0b 1111_0010 (8 bits)

ZExt(Lo) = 0b 0000_0000 0110_1001  (16 bits)
ZExt(Hi) = 0b 0000_0000 1111_0010  (16 bits)
Hi << 8  = 0b 1111_0010 0000_0000  (16 bits)

Result = Lo | (Hi << 8)
       = 0b 1111_0010 0110_1001    (16 bits, halves merged!)
```

### Why This Works for Vector Types

On AArch64, these operations map to efficient NEON instructions:

| Operation | NEON Instruction | Description |
|-----------|-----------------|-------------|
| TRUNCATE v8i16 → v8i8 | `XTN` | Narrow: keep lower half of each element |
| SRL v8i16, 8 | `USHR` | Unsigned shift right |
| ZERO_EXTEND v8i8 → v8i16 | `USHLL` | Unsigned shift left long (with shift=0) |
| SHL v8i16, 8 | `SHL` | Shift left |
| OR v8i16 | `ORR` | Bitwise OR |

Each of these is a **single instruction** operating on all lanes simultaneously. So the overhead of splitting, computing, and merging is only ~10 extra instructions — far less than the ~70 saved by avoiding bit-by-bit expansion.

---

## 7. Our Solution: Three Generic Strategies

We added three strategies to `expandCLMUL()` in `TargetLowering.cpp`, tried in order:

### Strategy 1: Karatsuba (Halve Element Width)

**When it applies**: Element width ≥ 16 bits, and the half-width vector type is legal and has (or can reach) legal CLMUL.

**Example**: `CLMUL(v8i16)` on AArch64
```
v8i16 → Karatsuba → 3×CLMUL(v8i8) + 1×CLMULH(v8i8)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                     All use PMUL (legal!)
```

**The code**:
```cpp
// Strategy 1: Karatsuba decomposition to half-element-width CLMUL.
unsigned HalfBW = BW / 2;
if (HalfBW >= 8) {
    EVT HalfVT = EVT::getVectorVT(Ctx, IntVT(HalfBW), VT.getVectorElementCount());
    if (isTypeLegal(HalfVT) && canNarrowCLMULToLegal(*this, Ctx, HalfVT)) {
        // Split X and Y into half-width elements
        // Compute Lo, Hi via Karatsuba identity
        // Merge and return
    }
}
```

### Strategy 2: Element Widen (Double Element Width)

**When it applies**: There's a double-width type where CLMUL is Legal or Custom.

**Example**: `CLMUL(v8i16)` on x86 with AVX2+PCLMUL
```
v8i16 → ZExt to v8i32 → CLMUL(v8i32) → Truncate back
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
         x86 has Custom CLMUL on v8i32 (via PCLMULQDQ)
```

**Why this works**: The CLMUL of i16 values zero-extended to i32 produces the correct i16 result in the lower bits — conceptually, `CLMUL(ZExt(X), ZExt(Y))` equals `ZExt(CLMUL(X, Y))` because the upper input bits are zero and don't contribute to the lower product bits.

### Strategy 3: Count Widen (Double Element Count)

**When it applies**: The current vector count is too narrow for a legal type, but doubling the count reaches one.

**Example**: `CLMUL(v4i16)` on AArch64
```
Problem: v4i16 → Karatsuba needs v4i8, but v4i8 is NOT a legal type (32-bit vector)

Solution:
  v4i16 → pad to v8i16 (insert into wider vector with undef padding)
       → Strategy 1 kicks in: Karatsuba to v8i8 (PMUL, legal!)
       → extract lower v4i16 from result
```

**Why this works**: CLMUL is element-wise. The upper lanes (filled with undef) don't affect the lower lanes. We just ignore them.

```
v4i16:  [ A₃ | A₂ | A₁ | A₀ ]
                                  INSERT_SUBVECTOR
v8i16:  [ undef | undef | undef | undef | A₃ | A₂ | A₁ | A₀ ]

After CLMUL:
v8i16:  [ garbage | garbage | ... | R₃ | R₂ | R₁ | R₀ ]
                                    ^^^^^^^^^^^^^^^^^^^^^^
                                    EXTRACT_SUBVECTOR → v4i16 result
```

### The Strategy Selection Helper: canNarrowCLMULToLegal()

This recursive function checks if a decomposition chain can reach legal CLMUL:

```
canNarrowCLMULToLegal(v4i32):
  → Is CLMUL(v4i32) legal? No.
  → Is v4i32 a legal type? Yes.
  → Try Karatsuba: v4i16
    → Is CLMUL(v4i16) legal? No.
    → Is v4i16 a legal type? Yes.
    → Try Karatsuba: v4i8
      → Is v4i8 a legal type? No. ← Dead end
    → Try widen: v8i16
      → Is CLMUL(v8i16) legal? No.
      → Is v8i16 a legal type? Yes.
      → Try Karatsuba: v8i8
        → Is CLMUL(v8i8) legal? YES! ← Found it!
      → Return true
    → Return true
  → Return true
```

The full chain for v4i32: `v4i32 → v4i16 → v8i16 → v8i8 (PMUL)`.

---

## 8. The Circular Expansion Trap

### The Problem

CLMULH has two expansion strategies:

**Path A (bitreverse):** `CLMULH(X, Y) = bitreverse(CLMUL(bitreverse(X), bitreverse(Y))) >> 1`
**Path B (widen):** `CLMULH(X, Y) = Truncate(CLMUL(ZExt(X), ZExt(Y)) >> BW)`

What happens if we're not careful with Path B for `CLMULH(v4i16)`?

```
CLMULH(v4i16)
  → Path B: ZExt to v4i32, do CLMUL(v4i32)
    → CLMUL(v4i32) needs Karatsuba decomposition
      → Karatsuba needs CLMULH(v4i16) ← CIRCULAR! We're back where we started!

CLMULH(v4i16) → CLMUL(v4i32) → Karatsuba → CLMULH(v4i16) → CLMUL(v4i32) → ...
     ↑                                              │
     └──────────── INFINITE LOOP! ──────────────────┘
```

### The Solution

We added `canNarrowCLMULToLegal()` as a guard on the CLMULH expansion path:

```cpp
// In the CLMULH case:
if (/* ... existing checks ... */ ||
    canNarrowCLMULToLegal(*this, Ctx, VT)) {
    // Use Path A (bitreverse): creates CLMUL(VT) which Strategy 1/3 can handle
} else {
    // Use Path B (widen): creates CLMUL(ExtVT)
}
```

When `canNarrowCLMULToLegal(v4i16)` is true, it means our strategies can efficiently expand CLMUL(v4i16). So we prefer Path A, which creates `CLMUL(v4i16)` — and that gets decomposed via Karatsuba to reach PMUL. No cycle.

### Visual: The Safe Path

```
CLMULH(v4i16)
  → Path A (bitreverse): BITREVERSE → CLMUL(v4i16) → BITREVERSE → SRL
                                         │
                                         ▼
                          Strategy 3: pad to v8i16
                                         │
                                         ▼
                          Strategy 1: Karatsuba to v8i8
                                         │
                                         ▼
                                   PMUL (legal!) ✅
```

No cycle — CLMUL expansion never creates CLMULH on a type we're trying to expand.

---

## 9. The BITREVERSE Bottleneck Fix

### The Discovery

After implementing the Karatsuba strategies, `v4i32 CLMUL` dropped from ~130 to ~131 instructions... almost no improvement! The bottleneck: **BITREVERSE on v4i16 and v8i16**.

The CLMULH expansion via the bitreverse path needs `BITREVERSE(v4i16)`. But AArch64 didn't have Custom lowering for v4i16/v8i16 BITREVERSE. It fell back to the **generic bit-manipulation expansion**: ~20+ instructions per BITREVERSE call.

### AArch64's BITREVERSE Hardware

AArch64 NEON has:
- `RBIT` — reverses all bits in each **byte** (v8i8 / v16i8 only)
- `REV16` — reverses the **byte order** within 16-bit elements
- `REV32` — reverses byte order within 32-bit elements
- `REV64` — reverses byte order within 64-bit elements

To do BITREVERSE on wider elements, combine byte-reversal + bit-reversal:

```
BITREVERSE(v4i16):
  Step 1: REV16 — reverse bytes within each 16-bit element
          [0xAB, 0xCD] → [0xCD, 0xAB]   (byte-swap within i16)

  Step 2: RBIT — reverse bits in each byte
          [0xCD, 0xAB] → [0xB3, 0xD5]   (bit-reverse each byte)

  Combined effect: full bit-reversal of each 16-bit element ✅
```

### What We Added

```cpp
// In AArch64ISelLowering.cpp:

// Declare v4i16/v8i16 BITREVERSE as Custom (previously only v2i32+ were Custom)
setOperationAction(ISD::BITREVERSE, MVT::v4i16, Custom);
setOperationAction(ISD::BITREVERSE, MVT::v8i16, Custom);

// In LowerBitreverse():
case MVT::v4i16:
    VST = MVT::v8i8;
    REVB = DAG.getNode(AArch64ISD::REV16, DL, VST, Op.getOperand(0));
    break;

case MVT::v8i16:
    VST = MVT::v16i8;
    REVB = DAG.getNode(AArch64ISD::REV16, DL, VST, Op.getOperand(0));
    break;
```

This generates just 2 instructions (REV16 + RBIT) instead of ~20+ generic bit manipulations.

### Impact

After the BITREVERSE fix:
- `v4i32 CLMUL`: 131 → **86** instructions
- `v4i32 CLMULH`: ~131 → **93** instructions

Still not as compact as v8i16 (because v4i32 goes through two Karatsuba levels), but a significant improvement.

---

## 10. Results: Before vs After

### AArch64 NEON Instruction Counts

| Function | Type | Before | After | Reduction |
|----------|------|--------|-------|-----------|
| CLMUL | v8i16 | ~70 | **18** | 74% |
| CLMUL | v4i16 | ~70 | **21** | 70% |
| CLMUL | v4i32 | ~130 | **86** | 34% |
| CLMUL | v2i32 | ~130 | **89** | 32% |
| CLMULR | v8i16 | ~72 | **24** | 67% |
| CLMULR | v4i16 | ~72 | **24** | 67% |
| CLMULR | v4i32 | ~130 | **92** | 29% |
| CLMULH | v8i16 | ~72 | **25** | 65% |
| CLMULH | v4i16 | ~72 | **25** | 65% |
| CLMULH | v4i32 | ~130 | **93** | 28% |

### Why v8i16 Is Best

v8i16 goes through **one** Karatsuba level: `v8i16 → v8i8 (PMUL)`. This is the sweet spot — minimal overhead.

### Why v4i32 Is Still Expensive

v4i32 goes through **multiple** decomposition levels:
```
v4i32 → Karatsuba → v4i16 → widen → v8i16 → Karatsuba → v8i8 (PMUL)
```

Each Karatsuba level creates 3 CLMULs + 1 CLMULH. Two levels means ~16 PMUL instructions plus all the splitting/merging/bitreverse overhead.

### x86 Also Benefits

Our Strategy 2 (element widen) automatically helps x86:

- `CLMUL(v8i16)` on AVX2+PCLMUL: now ZExts to `v8i32` where x86 has Custom CLMUL
- `CLMUL(v16i16)` on AVX512+VPCLMULQDQ: similar benefit

No x86-specific code was needed — the generic framework in `expandCLMUL()` handles it.

### Test Results

| Target | Tests | Pass | Fail | Notes |
|--------|-------|------|------|-------|
| AArch64 | 3920 | 3914 | 2 | Pre-existing failures |
| x86 | 5397 | 5380 | 0 | 15 expectedly failed, 2 unsupported |

---

## 11. The Stretch Goal: Scalar CLMUL via FPU Transfer

### What the Issue Says

> "A stretch goal would investigate whether scalar clmul expansion would benefit from transferring to/from the FPU to process as the lowest element of a v8i8/v4i16/v2i32/v1i64 type."

### The Idea

Currently, a scalar `CLMUL(i8, i8)` without hardware support is expanded bit-by-bit: ~8 iterations of shifts, AND, select, XOR = **~30+ instructions**.

But AArch64 has PMUL on v8i8! If we could:
1. Move the two scalar i8 values into a NEON register (FMOV)
2. Do a single PMUL instruction
3. Move the result back (FMOV)

...that would be just **3 instructions** instead of 30+.

### How It Would Work

```
Scalar CLMUL(i8 %a, i8 %b):

Before (bit-by-bit):
                                    ~30+ instructions of shifts/and/xor

After (FPU transfer):
    fmov  d0, x0          // Move scalar %a into NEON register (1 instr)
    fmov  d1, x1          // Move scalar %b into NEON register (1 instr)
    pmul  v0.8b, v0.8b, v1.8b  // Carryless multiply (1 instr, operates on lowest byte)
    fmov  x0, d0          // Move result back to GPR (1 instr)
                                    Total: 4 instructions (~87% reduction)
```

The key insight: PMUL operates element-wise on all 8 bytes. We only care about the lowest byte — the other 7 lanes produce garbage, but we ignore them.

### Why We Didn't Implement It

1. **It's explicitly marked as a "stretch goal"** — not required for the core task
2. **It requires a new strategy**: detecting scalar CLMUL and emitting `INSERT_VECTOR_ELT → CLMUL → EXTRACT_VECTOR_ELT`
3. **Cost model concerns**: The FPU transfer cost (FMOV scalar ↔ NEON) varies by microarchitecture. On some cores, crossing between integer and NEON pipelines adds latency
4. **It's a separate patch**: Orthogonal to the vector Karatsuba work

### When Would It Be Worth Doing?

- When scalar CLMUL appears in hot loops (CRC computation on byte arrays)
- On microarchitectures where NEON transfer is cheap (most modern AArch64 cores)
- Could also help i16, i32 scalar CLMUL via Karatsuba in NEON registers

---

## 12. Key Takeaways

### 1. Reuse Existing Mathematical Identities
The Karatsuba identity was already in `ExpandIntRes_CLMUL` for scalar type splitting. We reused the same math for vector element-width decomposition. Look for existing patterns before inventing new ones.

### 2. Generic Solutions Beat Target-Specific Ones
By implementing the strategies in `TargetLowering.cpp` (generic), all targets automatically benefit. x86 picked up improvements without a single line of x86 code.

### 3. Recursive Decomposition Needs Cycle Prevention
When operations expand into sub-operations that might circularly require the original operation, you need explicit guards. Our `canNarrowCLMULToLegal()` helper prevents the CLMULH → CLMUL → Karatsuba → CLMULH cycle.

### 4. Bottlenecks Can Be Surprising
The initial Karatsuba implementation barely helped v4i32 because **BITREVERSE** (not CLMUL) was the bottleneck. Always profile/measure — the slowest part may not be where you expect.

### 5. Legal Types ≠ Legal Operations
v8i16 is a **legal type** on AArch64 (fits in a Q register), but CLMUL on v8i16 is **not a legal operation** (no PMUL for 16-bit elements). This distinction is fundamental to LLVM's legalization system.

### 6. Multiple Strategies Can Coexist
Strategy 1 (Karatsuba), 2 (element widen), and 3 (count widen) are tried in order. Different strategies fire for different target/type combinations. This is more robust than a single approach.

---

## 13. Relationship to Previous Bugs

This is the **third** in a series of CLMUL-related changes:

### Bug 1: DAGCombiner Infinite Loop
**File**: `DAGCombiner.cpp`
**Problem**: DAGCombiner folds `bitreverse(clmul(bitreverse, bitreverse))` → CLMULR, then TargetLowering expands CLMULR back to the same pattern. Infinite loop.
**Fix**: Added legality check before folding.
**Learning doc**: `LEARNING_DAGCOMBINER_BUG.md`

### Bug 2: v8i8 CLMULR/CLMULH Expansion
**File**: `TargetLowering.cpp`
**Problem**: `expandCLMUL()` chose the zero-extend path for v8i8 CLMULR/CLMULH, creating CLMUL on v8i16 (not legal), resulting in 42 instructions instead of 4.
**Fix**: Added check for CLMUL legality on the extended type.
**Learning doc**: `LEARNING_V8I8_CLMULR_EXPANSION.md`

### This Change: Karatsuba Decomposition for Wide Vector CLMUL
**Files**: `TargetLowering.cpp`, `AArch64ISelLowering.cpp`
**Problem**: CLMUL on v8i16, v4i32, etc. expanded bit-by-bit (~70-130 instructions) despite legal v8i8 CLMUL being available.
**Fix**: Added Karatsuba decomposition, element widening, and count widening strategies.
**Learning doc**: This document (`LEARNING_VXI16_CLMUL_KARATSUBA.md`)

### The Connection

All three changes are in the **CLMUL expansion pipeline**. Each fixes a different aspect:

```
                   Bug 1: Infinite loop
                   (DAGCombiner ↔ TargetLowering cycle)
                          │
  IR with CLMUL patterns  │
          │                ▼
          ├──→ DAGCombiner  ──→  folds to CLMUL/CLMULR/CLMULH
          │                         │
          │                         ▼
          │              Operation Legalization
          │                         │
          │    Bug 2: Wrong path    │   This change: Missing
          │    for v8i8 CLMULR      │   Karatsuba strategies
          │         │               │         │
          │         ▼               ▼         ▼
          │    expandCLMUL() ───────────────────
          │         │
          │         ▼
          │    Legal PMUL instructions (v8i8, v16i8)
          │
          ▼
    Final machine code
```

Each change improved a different failure mode of the same expansion function, progressively making CLMUL codegen better across the board.
