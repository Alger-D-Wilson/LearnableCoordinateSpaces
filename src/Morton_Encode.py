import triton
import triton.language as tl

@triton.jit
def morton_encode_kernel(ix_ptr, iy_ptr, iz_ptr, out_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    ix = tl.load(ix_ptr + offs, mask=mask, other=0).to(tl.uint64)
    iy = tl.load(iy_ptr + offs, mask=mask, other=0).to(tl.uint64)
    iz = tl.load(iz_ptr + offs, mask=mask, other=0).to(tl.uint64)

    # --- expand_bits inline ---
    v = ix
    v = (v | (v << 32)) & 0x1f00000000ffff
    v = (v | (v << 16)) & 0x1f0000ff0000ff
    v = (v | (v << 8))  & 0x100f00f00f00f00f
    v = (v | (v << 4))  & 0x10c30c30c30c30c3
    v = (v | (v << 2))  & 0x1249249249249249
    mx = v

    v = iy
    v = (v | (v << 32)) & 0x1f00000000ffff
    v = (v | (v << 16)) & 0x1f0000ff0000ff
    v = (v | (v << 8))  & 0x100f00f00f00f00f
    v = (v | (v << 4))  & 0x10c30c30c30c30c3
    v = (v | (v << 2))  & 0x1249249249249249
    my = v

    v = iz
    v = (v | (v << 32)) & 0x1f00000000ffff
    v = (v | (v << 16)) & 0x1f0000ff0000ff
    v = (v | (v << 8))  & 0x100f00f00f00f00f
    v = (v | (v << 4))  & 0x10c30c30c30c30c3
    v = (v | (v << 2))  & 0x1249249249249249
    mz = v

    morton = (mx << 2) | (my << 1) | mz

    tl.store(out_ptr + offs, morton, mask=mask)