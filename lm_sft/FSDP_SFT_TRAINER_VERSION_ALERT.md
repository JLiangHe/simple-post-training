# FSDP SFT Trainer Version Alert

## Issue Overview
The FSDP SFT trainer has two versions that can be toggled by commenting/uncommenting code blocks in `fsdp_sft_trainer.py` lines 219-244.

## Version Details

### Version 1 (Currently Active - Lines 219-236)
**Complex Flash Attention Handler**
- Loads model with `attn_implementation="eager"` first
- Moves model to GPU with proper meta tensor handling
- Enables Flash Attention 2.0 after GPU placement
- **Pros**: Perfectly handles Flash Attention initialization issues
- **Cons**: Can cause CUDA OOM with large datasets (workable up to ~100k samples)

### Version 2 (Commented Out - Lines 238-244)
**Simple Direct Loading**
- Directly loads model with `attn_implementation="flash_attention_2"`
- **Pros**: Memory efficient, safer for large datasets
- **Cons**: May produce Flash Attention warnings, ~3x slower training speed

## Usage Instructions
To switch between versions:
1. Comment out the active version (lines 219-236)
2. Uncomment the simple version (lines 238-244)

## Recommendation
- Use **Version 1** for datasets ≤100k samples when Flash Attention stability is critical
- Use **Version 2** for large datasets when memory efficiency is more important than speed