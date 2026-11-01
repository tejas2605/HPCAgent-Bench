
import torch
import triton
import triton.language as tl


@triton.jit
def stencil_3d_kernel(
    in_ptr,                     # Input grid
    out_ptr,                    # Output grid
    w_ptr,                      # Weights for each radius
    N,                          # Number of elements in each dimension
    R,                          # Radius of the stencil
    # N: tl.constexpr,            # Number of elements in each dimension
    # R: tl.constexpr,            # Radius of the stencil
    BLOCK_SIZE: tl.constexpr,   # Size of each block
):
    # Get program ID of the current block within the grid
    pid = tl.program_id(0)
    # Compute the offsets for each thread in the block
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Create a mask to ensure we don't go out of bounds.
    # All elements processed should be valid, with offsets < N^3.
    mask = offsets < N * N * N

    # Convert 1D offsets into 3D coordinates
    x = offsets // (N * N)
    y = (offsets // N) % N
    z = offsets % N

    # Get the center element's weight
    center = tl.load(w_ptr + R)
    # Accumulate the produce of center weight*center element for all N^3 elements
    # Generates an N^3 sized tensor of the center element's contribution to the output grid
    acc = tl.load(in_ptr + offsets, mask, other=0) * center

    # Process each radius
    for r in range(1, R + 1):
        w = tl.load(w_ptr + r - 1)

        # X direction offsets
        xm = tl.maximum(x - r, 0)
        xp = tl.minimum(x + r, N - 1)

        # Load the value
        val = tl.load(
            in_ptr + xm * N * N + y * N + z,
            mask, other=0
        )
        # MAC
        acc += w * val

        # Load the value and MAC
        val = tl.load(
            in_ptr + xp * N * N + y * N + z,
            mask, other=0
        )
        acc += w * val

        # Y direction
        ym = tl.maximum(y - r, 0)
        yp = tl.minimum(y + r, N - 1)

        # Load the value and MAC
        val = tl.load(
            in_ptr + x * N * N + ym * N + z,
            mask, other=0
        )
        acc += w * val

        # Load the value and MAC
        val = tl.load(
            in_ptr + x * N * N + yp * N + z,
            mask, other=0
        )
        acc += w * val

        # Z direction
        zm = tl.maximum(z - r, 0)
        zp = tl.minimum(z + r, N - 1)

        # Load the value and MAC
        val = tl.load(
            in_ptr + x * N * N + y * N + zm,
            mask, other=0
        )
        acc += w * val

        # Load the value and MAC
        val = tl.load(
            in_ptr + x * N * N + y * N + zp,
            mask, other=0
        )
        acc += w * val

    # Store output
    tl.store(out_ptr + offsets, acc, mask)


def stencil_3d(in_grid, w_dist, N, R):
    out_grid = torch.empty_like(in_grid)

    BLOCK_SIZE = 256
    grid = (triton.cdiv(N * N * N, BLOCK_SIZE),)

    stencil_3d_kernel[grid](
        in_grid,
        out_grid,
        w_dist,
        int(N),
        int(R),
        BLOCK_SIZE,
    )

    return out_grid
