package jflow.data;

import java.util.concurrent.ForkJoinPool;
import java.util.stream.IntStream;

class OptimizedMatmul {
    /**
     * Performs matrix multiplication.
     * 
     * @param matrixA The first matrix, dimensions [m, k]
     * @param matrixB The second matrix, dimensions [k, n]
     * @param m Number of rows in each A matrix
     * @param n Number of columns in each B matrix
     * @param k Number of columns in A / rows in B
     * @param scale Whether to scale each resulting by 1/sqrt(k)
     * @param BLOCK_SIZE_M Block size for dimension m
     * @param BLOCK_SIZE_N Block size for dimension n
     * @param BLOCK_SIZE_K Block size for dimension k
     * @param THREAD_POOL Thread pool for parallel execution
     * @return The result matrices, dimensions [m, n]
     */
    // NOTE: THIS FUNCTION IS FROM CLAUDE.AI 
    protected static float[] matmul(float[] matrixA, float[] matrixB, 
        int m, int n, int k, boolean scale, int BLOCK_SIZE_M, 
        int BLOCK_SIZE_N, int BLOCK_SIZE_K, ForkJoinPool THREAD_POOL) {
        
        float scaleFactor = scale ? (float)(1.0f / Math.sqrt(k)) : 1.0f;
        float[] result = new float[m * n];
        
        try {
            // Pre-transpose matrix B for better cache coherence if k and n are large enough
            final float[] matrixBTransposed;
            final boolean useTransposed = k > 256 && n > 256;
            
            if (useTransposed) {
                matrixBTransposed = new float[k * n];
                // Transpose B in parallel to improve memory access patterns
                transposeMat(matrixB, matrixBTransposed, k, n, THREAD_POOL);
            } else {
                matrixBTransposed = null; // Not used
            }
            
            // More granular task division to increase parallelism
            // Calculate total number of blocks across all dimensions
            int mBlocks = (m + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M;
            int nBlocks = (n + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N;
            int totalBlocks = mBlocks * nBlocks;
            
            // Submit work to reusable thread pool using a more fine-grained task distribution
            THREAD_POOL.submit(() -> {
                IntStream.range(0, totalBlocks).parallel().forEach(blockIdx -> {
                    // Convert flat index to 2D block coordinates
                    int blockI = blockIdx / nBlocks;
                    int blockJ = blockIdx % nBlocks;
                    
                    int jLimit = Math.min((blockJ + 1) * BLOCK_SIZE_N, n);
                    
                    // Zero out the result region for this block
                    for (int i = blockI * BLOCK_SIZE_M; i < Math.min((blockI + 1) * BLOCK_SIZE_M, m); i++) {
                        for (int j = blockJ * BLOCK_SIZE_N; j < jLimit; j++) {
                            result[i * n + j] = 0.0f;
                        }
                    }
                    
                    // Process blocks in k dimension
                    for (int blockK = 0; blockK < (k + BLOCK_SIZE_K - 1) / BLOCK_SIZE_K; blockK++) {
                        int kLimit = Math.min((blockK + 1) * BLOCK_SIZE_K, k);
                        int kStart = blockK * BLOCK_SIZE_K;
                        
                        // Process each row in the current M block
                        for (int i = blockI * BLOCK_SIZE_M; i < Math.min((blockI + 1) * BLOCK_SIZE_M, m); i++) {
                            // For each column in the current N block
                            for (int j = blockJ * BLOCK_SIZE_N; j < jLimit; j += 8) {
                                // Reset accumulators for better register usage
                                float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
                                float sum4 = 0.0f, sum5 = 0.0f, sum6 = 0.0f, sum7 = 0.0f;
                                
                                int jEnd = Math.min(j + 8, jLimit);
                                int cols = jEnd - j;
                                
                                if (useTransposed) {
                                    // Process with transposed B
                                    for (int kIndex = kStart; kIndex < kLimit; kIndex++) {
                                        float aVal = matrixA[i * k + kIndex];
                                        
                                        // Unrolled processing of 8 columns with transposed B
                                        if (cols > 0) sum0 += aVal * matrixBTransposed[j * k + kIndex];
                                        if (cols > 1) sum1 += aVal * matrixBTransposed[(j+1) * k + kIndex];
                                        if (cols > 2) sum2 += aVal * matrixBTransposed[(j+2) * k + kIndex];
                                        if (cols > 3) sum3 += aVal * matrixBTransposed[(j+3) * k + kIndex];
                                        if (cols > 4) sum4 += aVal * matrixBTransposed[(j+4) * k + kIndex];
                                        if (cols > 5) sum5 += aVal * matrixBTransposed[(j+5) * k + kIndex];
                                        if (cols > 6) sum6 += aVal * matrixBTransposed[(j+6) * k + kIndex];
                                        if (cols > 7) sum7 += aVal * matrixBTransposed[(j+7) * k + kIndex];
                                    }
                                } else {
                                    // Process with original B
                                    for (int kIndex = kStart; kIndex < kLimit; kIndex++) {
                                        float aVal = matrixA[i * k + kIndex];
                                        
                                        // Unrolled processing of 8 columns
                                        if (cols > 0) sum0 += aVal * matrixB[kIndex * n + j];
                                        if (cols > 1) sum1 += aVal * matrixB[kIndex * n + j + 1];
                                        if (cols > 2) sum2 += aVal * matrixB[kIndex * n + j + 2];
                                        if (cols > 3) sum3 += aVal * matrixB[kIndex * n + j + 3];
                                        if (cols > 4) sum4 += aVal * matrixB[kIndex * n + j + 4];
                                        if (cols > 5) sum5 += aVal * matrixB[kIndex * n + j + 5];
                                        if (cols > 6) sum6 += aVal * matrixB[kIndex * n + j + 6];
                                        if (cols > 7) sum7 += aVal * matrixB[kIndex * n + j + 7];
                                    }
                                }
                                
                                // Accumulate results with scaling
                                if (cols > 0) result[i * n + j] += sum0 * scaleFactor;
                                if (cols > 1) result[i * n + j + 1] += sum1 * scaleFactor;
                                if (cols > 2) result[i * n + j + 2] += sum2 * scaleFactor;
                                if (cols > 3) result[i * n + j + 3] += sum3 * scaleFactor;
                                if (cols > 4) result[i * n + j + 4] += sum4 * scaleFactor;
                                if (cols > 5) result[i * n + j + 5] += sum5 * scaleFactor;
                                if (cols > 6) result[i * n + j + 6] += sum6 * scaleFactor;
                                if (cols > 7) result[i * n + j + 7] += sum7 * scaleFactor;
                            }
                        }
                    }
                });
            }).get(); // Wait for completion
        } catch (Exception e) {
            throw new RuntimeException("Error during parallel matrix multiplication", e);
        }
        
        return result;
    }


    /**
     * Helper method to transpose matrix B for better cache behavior
     */
    // NOTE: THIS FUNCTION IS FROM CLAUDE.AI 
     private static void transposeMat(float[] src, float[] dst, int rows, int cols, ForkJoinPool THREAD_POOL) {
        int blockSize = 64; // Cache-friendly block size
        
        try {
            THREAD_POOL.submit(() -> {
                IntStream.range(0, (rows + blockSize - 1) / blockSize).parallel().forEach(blockRow -> {
                    for (int blockCol = 0; blockCol < (cols + blockSize - 1) / blockSize; blockCol++) {
                        int rowEnd = Math.min((blockRow + 1) * blockSize, rows);
                        int colEnd = Math.min((blockCol + 1) * blockSize, cols);
                        
                        for (int i = blockRow * blockSize; i < rowEnd; i++) {
                            for (int j = blockCol * blockSize; j < colEnd; j++) {
                                dst[j * rows + i] = src[i * cols + j];
                            }
                        }
                    }
                });
            }).get();
        } catch (Exception e) {
            throw new RuntimeException("Error during parallel matrix transposition", e);
        }
    }

    /**
     * Performs batch matrix multiplication operation.
     * For each batch, computes matrixA[b] * matrixB[b] for all batches b.
     * 
     * @param batchMatrixA The first batch of matrices, dimensions [batchSize, m, k]
     * @param batchMatrixB The second batch of matrices, dimensions [batchSize, k, n]
     * @param batchSize Number of matrices in the batch
     * @param m Number of rows in each A matrix
     * @param n Number of columns in each B matrix
     * @param k Number of columns in A / rows in B
     * @param scale Whether to scale each resulting itemf by 1/sqrt(k)
     * @param BLOCK_SIZE_M Block size for dimension m
     * @param BLOCK_SIZE_N Block size for dimension n
     * @param BLOCK_SIZE_K Block size for dimension k
     * @param THREAD_POOL Thread pool for parallel execution
     * @return The result matrices, dimensions [batchSize, m, n]
     */
    // NOTE: THIS FUNCTION IS FROM CLAUDE.AI 
    protected static float[] batchMatmul(float[] batchMatrixA, float[] batchMatrixB, 
                                        int batchSize, int m, int n, int k, boolean scale,
                                        int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, 
                                        ForkJoinPool THREAD_POOL) {
        
        float scaleFactor = scale ? (float)(1.0f / Math.sqrt(k)) : 1.0f;
        float[] result = new float[batchSize * m * n];
        
        try {
            // Determine if transposing B is beneficial
            final boolean useTransposed = k > 256 && n > 256;
            final float[] batchMatrixBTransposed;
            
            if (useTransposed) {
                // Pre-transpose all B matrices in the batch
                batchMatrixBTransposed = new float[batchSize * k * n];
                batchTransposeMat(batchMatrixB, batchMatrixBTransposed, batchSize, k, n, THREAD_POOL);
            } else {
                batchMatrixBTransposed = null;
            }
            
            // Calculate total number of work units
            int mBlocks = (m + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M;
            int nBlocks = (n + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N;
            int totalBlocksPerBatch = mBlocks * nBlocks;
            int totalBlocks = totalBlocksPerBatch * batchSize;
            
            // Submit work to thread pool using fine-grained task distribution
            THREAD_POOL.submit(() -> {
                IntStream.range(0, totalBlocks).parallel().forEach(blockIdx -> {
                    // Decompose block index into batch and 2D coordinates
                    int batchIdx = blockIdx / totalBlocksPerBatch;
                    int localBlockIdx = blockIdx % totalBlocksPerBatch;
                    int blockI = localBlockIdx / nBlocks;
                    int blockJ = localBlockIdx % nBlocks;
                    
                    // Calculate batch offsets for current batch
                    int batchOffsetA = batchIdx * m * k;
                    int batchOffsetB = batchIdx * k * n;
                    int batchOffsetResult = batchIdx * m * n;
                    
                    int jLimit = Math.min((blockJ + 1) * BLOCK_SIZE_N, n);
                    
                    // Zero out the result region for this block
                    for (int i = blockI * BLOCK_SIZE_M; i < Math.min((blockI + 1) * BLOCK_SIZE_M, m); i++) {
                        for (int j = blockJ * BLOCK_SIZE_N; j < jLimit; j++) {
                            result[batchOffsetResult + i * n + j] = 0.0f;
                        }
                    }
                    
                    // Process blocks in k dimension
                    for (int blockK = 0; blockK < (k + BLOCK_SIZE_K - 1) / BLOCK_SIZE_K; blockK++) {
                        int kLimit = Math.min((blockK + 1) * BLOCK_SIZE_K, k);
                        int kStart = blockK * BLOCK_SIZE_K;
                        
                        // Process each row in the current M block
                        for (int i = blockI * BLOCK_SIZE_M; i < Math.min((blockI + 1) * BLOCK_SIZE_M, m); i++) {
                            // For each column in the current N block
                            for (int j = blockJ * BLOCK_SIZE_N; j < jLimit; j += 8) {
                                // Reset accumulators for better register usage
                                float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
                                float sum4 = 0.0f, sum5 = 0.0f, sum6 = 0.0f, sum7 = 0.0f;
                                
                                int jEnd = Math.min(j + 8, jLimit);
                                int cols = jEnd - j;
                                
                                if (useTransposed) {
                                    // Process with transposed B
                                    int batchOffsetBTransposed = batchIdx * k * n;
                                    
                                    for (int kIndex = kStart; kIndex < kLimit; kIndex++) {
                                        float aVal = batchMatrixA[batchOffsetA + i * k + kIndex];
                                        
                                        // Unrolled processing of 8 columns with transposed B
                                        if (cols > 0) sum0 += aVal * batchMatrixBTransposed[batchOffsetBTransposed + j * k + kIndex];
                                        if (cols > 1) sum1 += aVal * batchMatrixBTransposed[batchOffsetBTransposed + (j+1) * k + kIndex];
                                        if (cols > 2) sum2 += aVal * batchMatrixBTransposed[batchOffsetBTransposed + (j+2) * k + kIndex];
                                        if (cols > 3) sum3 += aVal * batchMatrixBTransposed[batchOffsetBTransposed + (j+3) * k + kIndex];
                                        if (cols > 4) sum4 += aVal * batchMatrixBTransposed[batchOffsetBTransposed + (j+4) * k + kIndex];
                                        if (cols > 5) sum5 += aVal * batchMatrixBTransposed[batchOffsetBTransposed + (j+5) * k + kIndex];
                                        if (cols > 6) sum6 += aVal * batchMatrixBTransposed[batchOffsetBTransposed + (j+6) * k + kIndex];
                                        if (cols > 7) sum7 += aVal * batchMatrixBTransposed[batchOffsetBTransposed + (j+7) * k + kIndex];
                                    }
                                } else {
                                    // Process with original B
                                    for (int kIndex = kStart; kIndex < kLimit; kIndex++) {
                                        float aVal = batchMatrixA[batchOffsetA + i * k + kIndex];
                                        
                                        // Unrolled processing of 8 columns
                                        if (cols > 0) sum0 += aVal * batchMatrixB[batchOffsetB + kIndex * n + j];
                                        if (cols > 1) sum1 += aVal * batchMatrixB[batchOffsetB + kIndex * n + j + 1];
                                        if (cols > 2) sum2 += aVal * batchMatrixB[batchOffsetB + kIndex * n + j + 2];
                                        if (cols > 3) sum3 += aVal * batchMatrixB[batchOffsetB + kIndex * n + j + 3];
                                        if (cols > 4) sum4 += aVal * batchMatrixB[batchOffsetB + kIndex * n + j + 4];
                                        if (cols > 5) sum5 += aVal * batchMatrixB[batchOffsetB + kIndex * n + j + 5];
                                        if (cols > 6) sum6 += aVal * batchMatrixB[batchOffsetB + kIndex * n + j + 6];
                                        if (cols > 7) sum7 += aVal * batchMatrixB[batchOffsetB + kIndex * n + j + 7];
                                    }
                                }
                                
                                // Accumulate results with scaling
                                if (cols > 0) result[batchOffsetResult + i * n + j] += sum0 * scaleFactor;
                                if (cols > 1) result[batchOffsetResult + i * n + j + 1] += sum1 * scaleFactor;
                                if (cols > 2) result[batchOffsetResult + i * n + j + 2] += sum2 * scaleFactor;
                                if (cols > 3) result[batchOffsetResult + i * n + j + 3] += sum3 * scaleFactor;
                                if (cols > 4) result[batchOffsetResult + i * n + j + 4] += sum4 * scaleFactor;
                                if (cols > 5) result[batchOffsetResult + i * n + j + 5] += sum5 * scaleFactor;
                                if (cols > 6) result[batchOffsetResult + i * n + j + 6] += sum6 * scaleFactor;
                                if (cols > 7) result[batchOffsetResult + i * n + j + 7] += sum7 * scaleFactor;
                            }
                        }
                    }
                });
            }).get(); // Wait for completion
        } catch (Exception e) {
            throw new RuntimeException("Error during batch matrix multiplication", e);
        }
        
        return result;
    }

    /**
     * Helper method to transpose all matrices in a batch
     */
    // NOTE: THIS FUNCTION IS FROM CLAUDE.AI 
    private static void batchTransposeMat(float[] src, float[] dst, int batchSize, int rows, int cols, ForkJoinPool THREAD_POOL) {
        int blockSize = 64; // Cache-friendly block size
        
        try {
            // Calculate total work units
            int rowBlocks = (rows + blockSize - 1) / blockSize;
            int colBlocks = (cols + blockSize - 1) / blockSize;
            int totalBlocksPerMatrix = rowBlocks * colBlocks;
            int totalBlocks = totalBlocksPerMatrix * batchSize;
            
            THREAD_POOL.submit(() -> {
                IntStream.range(0, totalBlocks).parallel().forEach(blockIdx -> {
                    // Decompose block index
                    int batchIdx = blockIdx / totalBlocksPerMatrix;
                    int localBlockIdx = blockIdx % totalBlocksPerMatrix;
                    int blockRow = localBlockIdx / colBlocks;
                    int blockCol = localBlockIdx % colBlocks;
                    
                    // Calculate batch offsets
                    int batchOffsetSrc = batchIdx * rows * cols;
                    int batchOffsetDst = batchIdx * rows * cols;
                    
                    int rowEnd = Math.min((blockRow + 1) * blockSize, rows);
                    int colEnd = Math.min((blockCol + 1) * blockSize, cols);
                    
                    for (int i = blockRow * blockSize; i < rowEnd; i++) {
                        for (int j = blockCol * blockSize; j < colEnd; j++) {
                            dst[batchOffsetDst + j * rows + i] = src[batchOffsetSrc + i * cols + j];
                        }
                    }
                });
            }).get();
        } catch (Exception e) {
            throw new RuntimeException("Error during batch matrix transposition", e);
        }
    }

    
}
