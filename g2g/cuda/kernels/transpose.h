// Use 32 to match warp size for perfect coalescing
#define TILE_DIM 32
#define BLOCK_ROWS 8

template <class input_type>
__global__ void transpose(input_type* __restrict__ odata,
                          const input_type* __restrict__ idata, int width,
                          int height) {
  // Padding to avoid bank conflicts
  __shared__ input_type tile[TILE_DIM][TILE_DIM + 1];

  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;

  // READ: Load 32x32 tile from global into shared memory
  // The loop processes 4 rows per thread (BLOCK_ROWS=8, TILE_DIM=32)
  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
    if (x < width && (y + j) < height) {
      // Coalesced read: adjacent threads read adjacent addresses
      tile[threadIdx.y + j][threadIdx.x] = idata[(y + j) * width + x];
    }
  }

  __syncthreads();

  // WRITE: Transpose coordinates
  x = blockIdx.y * TILE_DIM + threadIdx.x;
  y = blockIdx.x * TILE_DIM + threadIdx.y;

  // Write 32x32 tile from shared to global memory
  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
    if (x < height && (y + j) < width) {
      // Coalesced write: adjacent threads write adjacent addresses
      odata[(y + j) * height + x] = tile[threadIdx.x][threadIdx.y + j];
    }
  }
}