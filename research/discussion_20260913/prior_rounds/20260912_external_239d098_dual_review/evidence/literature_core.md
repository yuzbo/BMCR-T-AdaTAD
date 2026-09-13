# Literature core audit (read-only web evidence, 2026-09-13)

## DyT (Dynamic Tuning), arXiv:2403.11808v2
FACT: §3.2 Eq.(2) states X' = Block(TD(X)) + Adapter(X): “Only the activated tokens are input into the Block, while all tokens are processed by the Adapter” (lines 120-122). Training uses Gumbel relaxation and says “during the fine-tuning stage, all tokens within X still need to traverse the Block” (lines 135-148). Inference feeds only K activated tokens to Block (149-151). §3.3 distinguishes Attention, MLP, Attention-MLP, Layer Dispatch; MLP dispatch preserves attention token interaction, attention dispatch may hurt it (155-169).
CODE: pinned segmentation file imports Adapter, TokenSelect (GitHub d1744f0 displayed line 1470). dynamic_adapter.py viewer exposed _gumbel_sigmoid implementation (display lines 517-555), but requested 26-76 were not stably exposed; exact code-range verification partial.
INFERENCE: claim is accurate for MLP-dispatch reading, but paper allows Block=Attn/MLP/entire layer. Full training Block compute means downstream optimization sees dense outputs plus adapter; inference sees skipped/masked Block. MLP pointwise skip differs materially from attention skip.

## CoLT5 v3, arXiv:2303.09752v3
FACT: §3.1 splits FFN and attention into light all-token and heavy selected-token branches; light FFN lower hidden dimension, heavy higher; light attention local/fewer heads, heavy full attention over separately selected tokens (lines 71-74). Three independent routers route FFN, attention queries, and attention key-values (124-125). Conditional attention separately routes q and kv; branches differ in heads/context (149-163). CoDA comparison states CoLT5 is pretrained from scratch and all parameters finetuned downstream (CoDA HTML lines 92-93), hence not dense fitting of a frozen pretrained model with small adapter.
INFERENCE: operator-specific routing matters: FFN is pointwise; attention changes cross-token context. Training regime limits direct adapter-only transfer.

## CoDA, arXiv:2304.04947v2
FACT: Abstract says it starts from an existing dense pretrained model and adds sparse activation plus few new parameters (lines 63-65). Pretrained blocks fixed; adapter updated; selected tokens use heavy block and others skip (74-80). Adapter applies all tokens while k tokens enter conditional branch (101-114); Y = X + Z_adapter + m⊙Z_cond (130-137). Downstream updates only adapter, router, layer norm (156-161). CoLT5 contrast at 92-93.
INFERENCE: closest evidence for conditional-path adaptation, but pointwise adapter all-token and routed frozen Transformer are distinct; it supports testing mixed-state exposure, not equivalence of dense new-model training.

## DToP, arXiv:2308.01045v2
FACT: §1 early-finalizes easy tokens, continues hard tokens; stage outputs jointly form result (55-60). Retains k highest-confidence tokens per semantic category for context (60,67-68). §3 uses auxiliary heads/stages; high-confidence tokens finalized, low-confidence continue (88-105). This is early-exit/pruning with positional outputs/context retention, not simple per-token MLP masking.
LIMIT: requested README commit 2baefffc879622b6514504dff44cf04d7e0e7504 was unreachable (GitHub internal error); README-specific prune-adaptation wording unverified.
INFERENCE: dense prediction and attention/context semantics argue skipped-token states are not interchangeable with dense states.

## TR-BERT, NAACL 2021
FACT: ACL page identifies paper and links author PDF/source (lines 91-100); abstract metadata says token-level layer-number adaptation at inference to avoid redundant calculation (line 129). HTML does not expose method sections, so exact exit-state/original-position/router-training claims are unverified in this pass.
LIMIT: use linked author PDF/source for mechanics; crawl confirms only token-level layer-number adaptation.

## Cross-paper implication
DyT/CoDA establish dense training execution can coexist with masked/routed inference and all-token pointwise adapters. CoLT5 and DyT §3.3 establish attention routing is not equivalent to pointwise MLP masking because it changes cross-token context. Any “full training computation is enough” claim must distinguish MLP residual masking from attention skip/routing. Literature supports the mechanism distinction, not a final project recommendation.
