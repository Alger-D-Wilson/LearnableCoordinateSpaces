import triton
import triton.language as tl

@triton.jit
def morton_decode_kernel(morton_ptr, ix_ptr, iy_ptr, iz_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    morton = tl.load(morton_ptr + offs, mask=mask, other=0).to(tl.uint64)

    # --- decode ix ---
    v = morton >> 2
    v = v & 0x1249249249249249
    v = (v ^ (v >> 2))  & 0x10c30c30c30c30c3
    v = (v ^ (v >> 4))  & 0x100f00f00f00f00f
    v = (v ^ (v >> 8))  & 0x1f0000ff0000ff
    v = (v ^ (v >> 16)) & 0x1f00000000ffff
    v = (v ^ (v >> 32)) & 0x1fffff
    ix = v

    # --- decode iy ---
    v = morton >> 1
    v = v & 0x1249249249249249
    v = (v ^ (v >> 2))  & 0x10c30c30c30c30c3
    v = (v ^ (v >> 4))  & 0x100f00f00f00f00f
    v = (v ^ (v >> 8))  & 0x1f0000ff0000ff
    v = (v ^ (v >> 16)) & 0x1f00000000ffff
    v = (v ^ (v >> 32)) & 0x1fffff
    iy = v

    # --- decode iz ---
    v = morton
    v = v & 0x1249249249249249
    v = (v ^ (v >> 2))  & 0x10c30c30c30c30c3
    v = (v ^ (v >> 4))  & 0x100f00f00f00f00f
    v = (v ^ (v >> 8))  & 0x1f0000ff0000ff
    v = (v ^ (v >> 16)) & 0x1f00000000ffff
    v = (v ^ (v >> 32)) & 0x1fffff
    iz = v

    # store outputs
    tl.store(ix_ptr + offs, ix, mask=mask)
    tl.store(iy_ptr + offs, iy, mask=mask)
    tl.store(iz_ptr + offs, iz, mask=mask)