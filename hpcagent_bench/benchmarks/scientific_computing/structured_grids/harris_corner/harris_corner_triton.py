import torch
import triton
import triton.language as tl

@triton.jit
def harris_corner_kernel(
    img_ptr,    # Input Image matrix
    out_ptr,    # Output Harris Scores matrix
    H,          # Height of the image
    W,          # Width of the image
    
    # k = Sensitivity constant 
    # It distinguishes how the strongly the detector distinguishes between edges and corners
    k,          # Sensitivity constant
    BLOCK_SIZE: tl.constexpr,  # Block Size
):

    pid = tl.program_id(0)
    # Offsets for each value in each BLOCK
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Offsets for weights across y direction
    y = offsets // W
    # Offsets for weights across x direction
    x = offsets % W

    mask = (
        (offsets < H * W)           # If the offset is within the image's dimensions
        & (y >= 2) & (y < H - 2)    # 2 rows (Top and Bottom) are untouched
        & (x >= 2) & (x < W - 2)    # 2 columns (Left and Right) are untouched
    )

    # Initialize empty arrays for Sxx, Syy, Sxy
    Sxx = tl.full((BLOCK_SIZE,), 0.0, tl.float32)
    Syy = tl.full((BLOCK_SIZE,), 0.0, tl.float32)
    Sxy = tl.full((BLOCK_SIZE,), 0.0, tl.float32)

    for dy in tl.static_range(-1, 2):
        for dx in tl.static_range(-1, 2):

            gy = y + dy
            gx = x + dx

            # Sobel X gradient
            Ix = (
                tl.load(img_ptr + (gy - 1) * W + gx + 1, mask=mask, other=0.0)
                - tl.load(img_ptr + (gy - 1) * W + gx - 1, mask=mask, other=0.0)
                + 2.0 * (
                    tl.load(img_ptr + gy * W + gx + 1, mask=mask, other=0.0)
                    - tl.load(img_ptr + gy * W + gx - 1, mask=mask, other=0.0)
                )
                + tl.load(img_ptr + (gy + 1) * W + gx + 1, mask=mask, other=0.0)
                - tl.load(img_ptr + (gy + 1) * W + gx - 1, mask=mask, other=0.0)
            ) * 0.125

            # Sobel Y gradient
            Iy = (
                tl.load(img_ptr + (gy + 1) * W + gx - 1, mask=mask, other=0.0)
                - tl.load(img_ptr + (gy - 1) * W + gx - 1, mask=mask, other=0.0)
                + 2.0 * (
                    tl.load(img_ptr + (gy + 1) * W + gx, mask=mask, other=0.0)
                    - tl.load(img_ptr + (gy - 1) * W + gx, mask=mask, other=0.0)
                )
                + tl.load(img_ptr + (gy + 1) * W + gx + 1, mask=mask, other=0.0)
                - tl.load(img_ptr + (gy - 1) * W + gx + 1, mask=mask, other=0.0)
            ) * 0.125

            Sxx += Ix * Ix
            Syy += Iy * Iy
            Sxy += Ix * Iy

    # Determinant = Helps measure whether an intensity change has happened in both directions
    det = Sxx * Syy - Sxy * Sxy
    # Trace = Sum of diagonal elements, measures overall gradient strength 
    trace = Sxx + Syy

    # Actual Harris Response calculation
    response = det - k * trace * trace

    # Store results
    tl.store(out_ptr + offsets, response, mask)

def kernel(img, R, k=0.04):
    H, W = img.shape
    out = R.clone()

    BLOCK_SIZE = 256

    # Split the workload into "Image Size / 256" blocks
    grid = (triton.cdiv(H * W, BLOCK_SIZE),)

    harris_corner_kernel[grid](
        img,
        out,
        H,
        W,
        float(k),
        BLOCK_SIZE,
    )

    return out