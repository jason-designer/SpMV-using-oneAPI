//==============================================================
// This sample provides a parallel implementation of a merge based sparse matrix
// and vector multiplication algorithm using SYCL. The input matrix is in
// compressed sparse row format.
//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <CL/sycl.hpp>
#include <iostream>
#include <map>
#include <set>

// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities/<version>/include/dpc_common.hpp
#include "dpc_common.hpp"

using namespace std;
using namespace sycl;

// Compressed Sparse Row (CSR) representation for sparse matrix.
//
// Example: The following 4 x 4 sparse matrix
//
//   a 0 0 0
//   b c 0 0
//   0 0 0 d
//   0 0 e f
//
// have 6 non zero elements in it:
//
//   Index  Row  Column  Value
//       0    0       0      a
//       1    1       0      b
//       2    1       1      c
//       3    2       3      d
//       4    3       2      e
//       5    3       3      f
//
// Its CSR representation is have three components:
// - Nonzero values: a, b, c, d, e, f
// - Column indices: 0, 0, 1, 3, 2, 3
// - Row offsets: 0, 1, 3, 4, 6
//
// Non zero values and their column indices directly correspond to the entries
// in the above table.
//
// Row offsets are offsets in the values array for the first non zero element of
// each row of the matrix.
//
//   Row  NonZeros  NonZeros_SeenBefore
//     0         1                    0
//     1         2                    1
//     2         1                    3
//     3         2                    4
//     -         -                    6
typedef struct {
  int *row_offsets;
  int *column_indices;
  float *values;
} CompressedSparseRow;

// Allocate unified shared memory for storing matrix and vectors so that they
// are accessible from both the CPU and the device (e.g., a GPU).
bool AllocateMemory(queue &q, int thread_count, CompressedSparseRow *matrix, int n, int nonzero,
                    float **x, float **y_sequential, float **y_parallel, float **y_lightspmv,
                    int **carry_row, float **carry_value, int **row_counter_space, int **row, float **sum) {
  matrix->row_offsets = malloc_shared<int>(n + 1, q);
  matrix->column_indices = malloc_shared<int>(nonzero, q);
  matrix->values = malloc_shared<float>(nonzero, q);

  *x = malloc_shared<float>(n, q);
  *y_sequential = malloc_shared<float>(n, q);
  *y_parallel = malloc_shared<float>(n, q);
  *y_lightspmv = malloc_shared<float>(n, q);

  *carry_row = malloc_shared<int>(thread_count, q);
  *carry_value = malloc_shared<float>(thread_count, q);
    
  *row_counter_space = malloc_shared<int>(1, q);
  *row = malloc_shared<int>(thread_count, q);
  *sum = malloc_shared<float>(thread_count, q);

  return (matrix->row_offsets != nullptr) &&
         (matrix->column_indices != nullptr) && (matrix->values != nullptr) &&
         (*x != nullptr) && (*y_sequential != nullptr) &&
         (*y_parallel != nullptr) && (*y_lightspmv != nullptr) && 
         (*carry_row != nullptr) && (*carry_value != nullptr) &&
         (*row_counter_space != nullptr) && (*row != nullptr) && (*sum != nullptr);
}

// Free allocated unified shared memory.
void FreeMemory(queue &q, CompressedSparseRow *matrix, float *x,
                float *y_sequential, float *y_parallel, float *y_lightspmv, 
                int *carry_row, float *carry_value, int *row_counter_space, int *row, float *sum) {
  if (matrix->row_offsets != nullptr) free(matrix->row_offsets, q);
  if (matrix->column_indices != nullptr) free(matrix->column_indices, q);
  if (matrix->values != nullptr) free(matrix->values, q);

  if (x != nullptr) free(x, q);
  if (y_sequential != nullptr) free(y_sequential, q);
  if (y_parallel != nullptr) free(y_parallel, q);
  if (y_lightspmv != nullptr) free(y_lightspmv, q);
    
  if (carry_row != nullptr) free(carry_row, q);
  if (carry_value != nullptr) free(carry_value, q);
  if (row_counter_space != nullptr) free(row_counter_space, q);
  if (row != nullptr) free(row, q);
  if (sum != nullptr) free(sum, q);
}

// Initialize inputs: sparse matrix and vector.
void InitializeSparseMatrixAndVector(const CompressedSparseRow *matrix, float *x, const int n, const int nonzero, const int max_value) {
  map<int, set<int>> indices;

  // Randomly choose a set of elements (i.e., row and column pairs) of the
  // matrix. These elements will have non zero values.
  for (int k = 0; k < nonzero; k++) {
    int i = rand() % n;
    int j = rand() % n;

    if (indices.find(i) == indices.end()) {
      indices[i] = {j};

    } else if (indices[i].find(j) == indices[i].end()) {
      indices[i].insert(j);

    } else {
      k--;
    }
  }

  int offset = 0;

  // Randomly choose non zero values of the sparse matrix.
  for (int i = 0; i < n; i++) {
    matrix->row_offsets[i] = offset;

    if (indices.find(i) != indices.end()) {
      set<int> &cols = indices[i];

      for (auto it = cols.cbegin(); it != cols.cend(); ++it, ++offset) {
        matrix->column_indices[offset] = *it;
        matrix->values[offset] = rand() % max_value + 1;
      }
    }
  }

  matrix->row_offsets[n] = nonzero;

  // Initialize input vector.
  for (int i = 0; i < n; i++) {
    x[i] = 1;
  }
}

// A sequential implementation of merge based sparse matrix and vector
// multiplication algorithm.
//
// Both row offsets and values indices can be thought of as sorted arrays. The
// progression of the computation is similar to that of merging two sorted
// arrays at a conceptual level.
//
// When a row offset and an index of the values array are equal (denoted as '?'
// below), the algorithm starts computing the value of a new element of the
// result vector.
//
// The algorithm continues to accumulate for the same element of the result
// vector otherwise (denoted as '*' below).
//
// Row indices ->  0 1 2 3
// Row offsets ->  0 1 3 4 6
//
//                 ?         0  a
//                   ?       1  b
//                   *       2  c
//                     ?     3  d
//                       ?   4  e
//                       *   5  f
//
//                           ^  ^
//                           |  |
//                           |  Non zero values
//                           |
//                           Indices of values array
void MergeSparseMatrixVector(CompressedSparseRow *matrix, int n, int nonzero, float *x, float *y) {
  int row_index = 0;
  int val_index = 0;

  y[row_index] = 0;

  while (val_index < nonzero) {
    if (val_index < matrix->row_offsets[row_index + 1]) {
      // Accumulate and move down.
      y[row_index] +=
          matrix->values[val_index] * x[matrix->column_indices[val_index]];
      val_index++;

    } else {
      // Move right.
      row_index++;
      y[row_index] = 0;
    }
  }

  for (row_index++; row_index < n; row_index++) {
    y[row_index] = 0;
  }
}

// Merge Coordinate.
typedef struct {
  int row_index;
  int val_index;
} MergeCoordinate;

// Given linear position on the merge path, find two dimensional merge
// coordinate (row index and value index pair) on the path.
MergeCoordinate MergePathBinarySearch(int diagonal, int *row_offsets, int n, int nonzero) {
  // Diagonal search range (in row index space).
  int row_min = (diagonal - nonzero > 0) ? (diagonal - nonzero) : 0;
  int row_max = (diagonal < n) ? diagonal : n;

  // 2D binary search along the diagonal search range.
  while (row_min < row_max) {
    int pivot = (row_min + row_max) >> 1;

    if (row_offsets[pivot + 1] <= diagonal - pivot - 1) {
      // Keep top right half of diagonal range.
      row_min = pivot + 1;
    } else {
      // Keep bottom left half of diagonal range.
      row_max = pivot;
    }
  }

  MergeCoordinate coordinate;

  coordinate.row_index = (row_min < n) ? row_min : n;
  coordinate.val_index = diagonal - row_min;

  return coordinate;
}

// The parallel implementation of spare matrix, vector multiplication algorithm
// uses this function as a subroutine. Each available thread calls this function
// with identical inputs, except the thread identifier (TID) is unique. Having a
// unique TID, each thread independently identifies its own, non overlapping
// share of the overall work. More importantly, each thread, except possibly the
// last one, handles the same amount of work. This implementation is an
// extension of the sequential implementation of the merge based sparse matrix,
// vector multiplication algorithm. It first identifies its scope of the merge
// and then performs only the amount of work that belongs this thread in the
// cohort of threads.
void MergeSparseMatrixVectorThread(int thread_count, int tid, int n, int nonzero,
                                   CompressedSparseRow matrix, float *x,
                                   float *y, int *carry_row,
                                   float *carry_value) {
  int path_length = n + nonzero;  // Merge path length.
  int items_per_thread = (path_length + thread_count - 1) /
                         thread_count;  // Merge items per thread.

  // Find start and end merge path coordinates for this thread.
  int diagonal = ((items_per_thread * tid) < path_length)
                     ? (items_per_thread * tid)
                     : path_length;
  int diagonal_end = ((diagonal + items_per_thread) < path_length)
                         ? (diagonal + items_per_thread)
                         : path_length;

  MergeCoordinate path = MergePathBinarySearch(diagonal, matrix.row_offsets, n, nonzero);
  MergeCoordinate path_end =
      MergePathBinarySearch(diagonal_end, matrix.row_offsets, n, nonzero);

  // Consume items-per-thread merge items.
  float dot_product = 0;

  for (int i = 0; i < items_per_thread; i++) {
    if (path.val_index < matrix.row_offsets[path.row_index + 1]) {
      // Accumulate and move down.
      dot_product += matrix.values[path.val_index] *
                     x[matrix.column_indices[path.val_index]];
      path.val_index++;

    } else {
      // Output row total and move right.
      y[path.row_index] = dot_product;
      dot_product = 0;
      path.row_index++;
    }
  }

  // Save carry.
  carry_row[tid] = path_end.row_index;
  carry_value[tid] = dot_product;
}

// This is the parallel implementation of merge based sparse matrix and vector
// mutiplication algorithm. It works in three steps:
//   1. Initialize elements of the output vector to zero.
//   2. Multiply sparse matrix and vector.
//   3. Fix up rows of the output vector that spanned across multiple threads.
// First two steps are parallel. They utilize all available processors
// (threads). The last step performs a reduction. It could be parallel as well
// but is kept as sequential for the following reasons:
//   1. Number of operation in this step is proportional to the number of
//   processors (threads).
//   2. Number of available threads is not too high.
void MergeSparseMatrixVector(queue &q, int compute_units, int work_group_size, int n, int nonzero,
                             CompressedSparseRow matrix, float *x, float *y,
                             int *carry_row, float *carry_value) {
  int thread_count = compute_units * work_group_size;

  // Initialize output vector.
  q.parallel_for<class InitializeVector>(
      nd_range<1>(compute_units * work_group_size, work_group_size),
      [=](nd_item<1> item) {
        auto global_id = item.get_global_id(0);
        auto items_per_thread = (n + thread_count - 1) / thread_count;
        auto start = global_id * items_per_thread;
        auto stop = start + items_per_thread;

        for (auto i = start; (i < stop) && (i < n); i++) {
          y[i] = 0;
        }
      });

  q.wait();

  // Multiply sparse matrix and vector.
  q.parallel_for<class MergeCsrMatrixVector>(
      nd_range<1>(compute_units * work_group_size, work_group_size),
      [=](nd_item<1> item) {
        auto global_id = item.get_global_id(0);
        MergeSparseMatrixVectorThread(thread_count, global_id, n, nonzero, matrix, x, y, carry_row, carry_value);
      });

  q.wait();

  // Carry fix up for rows spanning multiple threads.
  for (int tid = 0; tid < thread_count - 1; tid++) {
    if (carry_row[tid] < n) {
      y[carry_row[tid]] += carry_value[tid];
    }
  }
}

// This is the implementation of LightSpmv
void LightSpmvKernel(queue &q, int compute_units, int work_group_size,
                    int *row_counter_space, int *row, float *sum, 
                    int vector_size, int numRows,
                    CompressedSparseRow matrix, float *x, float *y){

    int numThreads = compute_units * work_group_size;
    
    row_counter_space[0] = 0;
    
    q.submit([&](handler &h) {     
        h.parallel_for(nd_range<1>(compute_units * work_group_size, work_group_size), [=](nd_item<1> item)[[intel::reqd_sub_group_size(32)]]{
            // thread info
            auto sg = item.get_sub_group();
            auto tid = item.get_global_id(0);
            auto lane_id = tid & (vector_size - 1);
            auto vector_id = (tid & (32 - 1)) / vector_size;
            
            // define atomic varilable
            auto row_counter = atomic_ref<int, 
              sycl::memory_order::relaxed, 
              sycl::memory_scope::device, 
              access::address_space::global_space>(row_counter_space[0]);
            
            // get new row
            if(lane_id == 0) row[tid] = row_counter.fetch_add(1);                                   // 每个vector获取一个新的行号row
            for(int v = 0; v < 32 / vector_size; v++){                                              // 将获得的row广播给同一个vector的其他线程
                if(vector_id == v) row[tid] = select_from_group(sg, row[tid], v * vector_size);
            }      
                     
            while(row[tid] < numRows){                                                              // 当获得的row大于numRows时说明所有行计算完成，退出循环
                int nnz_start = matrix.row_offsets[row[tid]];
                int nnz_end = matrix.row_offsets[row[tid] + 1];
                sum[tid] = 0;
                
                // compute a row of matrix
                for(int i = nnz_start; i + lane_id < nnz_end; i += vector_size){
                    sum[tid] += matrix.values[i + lane_id] * x[matrix.column_indices[i + lane_id]]; // 将矩阵行的点积累加到vector中
                }
                    
                // get sum of vector
                for(int i = vector_size >> 1; i > 0; i = i >> 1){
                    sum[tid] += shift_group_left(sg, sum[tid], i);                                  // 将结果累加到vector的第一个线程
                }
                
                // write back
                if(lane_id == 0) y[row[tid]] = sum[tid];                                            // vector的第一个线程写回到y中
                
                // get new row
                if(lane_id == 0) row[tid] = row_counter.fetch_add(1);                               // 每个vector获取一个新的行号row
                for(int v = 0; v < 32 / vector_size; v++){                                          // 将获得的row广播给同一个vector的其他线程
                    if(vector_id == v) row[tid] = select_from_group(sg, row[tid], v * vector_size);
                }      
            }
        });
    }).wait();
}

void LightSpmv(queue &q, int compute_units, int work_group_size,
                int *row_counter_space, int *row, float *sum,
                int numRows, int nonzero, 
                CompressedSparseRow matrix, float *x, float *y){
    double mean = (double)nonzero / (double)numRows;
    if(mean < 16)                       LightSpmvKernel(q, compute_units, work_group_size, row_counter_space, row, sum, 2,  numRows, matrix, x, y);
    else if(16  <= mean && mean < 96)   LightSpmvKernel(q, compute_units, work_group_size, row_counter_space, row, sum, 4,  numRows, matrix, x, y);
    else if(96  <= mean && mean < 832)  LightSpmvKernel(q, compute_units, work_group_size, row_counter_space, row, sum, 8,  numRows, matrix, x, y);
    else if(832 <= mean && mean < 3096) LightSpmvKernel(q, compute_units, work_group_size, row_counter_space, row, sum, 16, numRows, matrix, x, y);
    else                                LightSpmvKernel(q, compute_units, work_group_size, row_counter_space, row, sum, 32, numRows, matrix, x, y);
}

// Check if two input vectors are equal.
bool VerifyVectorsAreEqual(float *u, float *v, int n) {
  for (int i = 0; i < n; i++) {
    if (fabs(u[i] - v[i]) > 1E-06) {
      return false;
    }
  }

  return true;
}


void exp_lightspmv(int n, int nonzero, int repetitions, int max_value, int compute_units, int work_group_size) {    
    // result
    double result = 0;

    // Sparse matrix.
    CompressedSparseRow matrix;

    // Input vector.
    float *x;

    // Vector: result of sparse matrix and vector multiplication.
    float *y_sequential;
    float *y_parallel;
    float *y_lightspmv;
    
    // Auxiliary storage for parallel computation.
    int *carry_row;
    float *carry_value;
    int *row_counter_space;
    int *row;
    float* sum;

    try {
        queue q{default_selector_v};

        // Allocate memory.
        if (!AllocateMemory(q, compute_units * work_group_size, &matrix, n, nonzero, &x, &y_sequential, &y_parallel, &y_lightspmv, &carry_row, &carry_value, &row_counter_space, &row, &sum)) {
          cout << "Memory allocation failure.\n";
          FreeMemory(q, &matrix, x, y_sequential, y_parallel, y_lightspmv, carry_row, carry_value, row_counter_space, row, sum);
          exit(-1);
        }

        // Initialize.
        InitializeSparseMatrixAndVector(&matrix, x, n, nonzero , max_value);
        
        

        // Warm up and Verify two results are equal.
        LightSpmvKernel(q, compute_units, work_group_size, row_counter_space, row, sum, 2, n, matrix, x, y_lightspmv);
        LightSpmvKernel(q, compute_units, work_group_size, row_counter_space, row, sum, 4, n, matrix, x, y_lightspmv);
        LightSpmvKernel(q, compute_units, work_group_size, row_counter_space, row, sum, 8, n, matrix, x, y_lightspmv);
        LightSpmvKernel(q, compute_units, work_group_size, row_counter_space, row, sum, 16, n, matrix, x, y_lightspmv);
        LightSpmvKernel(q, compute_units, work_group_size, row_counter_space, row, sum, 32, n, matrix, x, y_lightspmv);
        MergeSparseMatrixVector(&matrix, n, nonzero, x, y_sequential);
        if (!VerifyVectorsAreEqual(y_sequential, y_lightspmv, n)) {
            cout << "LightSpmv: Failed to correctly compute!\n";
            exit(-1);
        }

        // Time executions.
        double elapsed_2 = 0;
        double elapsed_4 = 0;
        double elapsed_8 = 0;
        double elapsed_16 = 0;
        double elapsed_32 = 0;
        
        for (int i = 0; i < repetitions; i++) {
            // LightSpmv compute
            dpc_common::TimeInterval timer_2;
            LightSpmvKernel(q, compute_units, work_group_size, row_counter_space, row, sum, 2, n, matrix, x, y_lightspmv);
            elapsed_2 += timer_2.Elapsed();
            
            dpc_common::TimeInterval timer_4;
            LightSpmvKernel(q, compute_units, work_group_size, row_counter_space, row, sum, 4, n, matrix, x, y_lightspmv);
            elapsed_4 += timer_4.Elapsed();
            
            dpc_common::TimeInterval timer_8;
            LightSpmvKernel(q, compute_units, work_group_size, row_counter_space, row, sum, 8, n, matrix, x, y_lightspmv);
            elapsed_8 += timer_8.Elapsed();
            
            dpc_common::TimeInterval timer_16;
            LightSpmvKernel(q, compute_units, work_group_size, row_counter_space, row, sum, 16, n, matrix, x, y_lightspmv);
            elapsed_16 += timer_16.Elapsed();
            
            dpc_common::TimeInterval timer_32;
            LightSpmvKernel(q, compute_units, work_group_size, row_counter_space, row, sum, 32, n, matrix, x, y_lightspmv);
            elapsed_32 += timer_32.Elapsed();
        }
        
        elapsed_2 /= repetitions;
        elapsed_4 /= repetitions;
        elapsed_8 /= repetitions;
        elapsed_16 /= repetitions;
        elapsed_32 /= repetitions;
        
        std::cout << "nonzero: " << nonzero << ", numRows: " << n <<  ", mean: " << nonzero / n << ", LightSpMV2: " << elapsed_2 << " sec." << "\n";
        std::cout << "nonzero: " << nonzero << ", numRows: " << n <<  ", mean: " << nonzero / n << ", LightSpMV4: " << elapsed_4 << " sec." << "\n";
        std::cout << "nonzero: " << nonzero << ", numRows: " << n <<  ", mean: " << nonzero / n << ", LightSpMV8: " << elapsed_8 << " sec." << "\n";
        std::cout << "nonzero: " << nonzero << ", numRows: " << n <<  ", mean: " << nonzero / n << ", LightSpMV16: " << elapsed_16 << " sec." << "\n";
        std::cout << "nonzero: " << nonzero << ", numRows: " << n <<  ", mean: " << nonzero / n << ", LightSpMV32: " << elapsed_32 << " sec." << "\n";

        FreeMemory(q, &matrix, x, y_sequential, y_parallel, y_lightspmv, carry_row, carry_value, row_counter_space, row, sum);  
        } catch (std::exception const &e) {
            cout << "An exception is caught while computing on device.\n";
            terminate();
    }
    return;
}


void exp_all(int n, int nonzero, int repetitions, int max_value, int compute_units, int work_group_size) {    
    // result
    double result = 0;

    // Sparse matrix.
    CompressedSparseRow matrix;

    // Input vector.
    float *x;

    // Vector: result of sparse matrix and vector multiplication.
    float *y_sequential;
    float *y_parallel;
    float *y_lightspmv;
    
    // Auxiliary storage for parallel computation.
    int *carry_row;
    float *carry_value;
    int *row_counter_space;
    int *row;
    float* sum;

    try {
        queue q{default_selector_v};

        // Allocate memory.
        if (!AllocateMemory(q, compute_units * work_group_size, &matrix, n, nonzero, &x, &y_sequential, &y_parallel, &y_lightspmv, &carry_row, &carry_value, &row_counter_space, &row, &sum)) {
          cout << "Memory allocation failure.\n";
          FreeMemory(q, &matrix, x, y_sequential, y_parallel, y_lightspmv, carry_row, carry_value, row_counter_space, row, sum);
          exit(-1);
        }

        // Initialize.
        InitializeSparseMatrixAndVector(&matrix, x, n, nonzero , max_value);

        // Warm up and Verify two results are equal.
        LightSpmv(q, compute_units, work_group_size, row_counter_space, row, sum, n, nonzero, matrix, x, y_lightspmv);
        MergeSparseMatrixVector(q, compute_units, work_group_size, n, nonzero, matrix, x, y_parallel, carry_row, carry_value);
        MergeSparseMatrixVector(&matrix, n, nonzero, x, y_sequential);
        if (!VerifyVectorsAreEqual(y_sequential, y_lightspmv, n)) {
            cout << "LightSpmv: Failed to correctly compute!\n";
            exit(-1);
        }
        if (!VerifyVectorsAreEqual(y_sequential, y_parallel, n)) {
            cout << "Merge-based: Failed to correctly compute!\n";
            exit(-1);
        }

        // Time executions.
        double elapsed_l = 0;
        double elapsed_m = 0;
        
        for (int i = 0; i < repetitions; i++) {
            // LightSpmv compute
            dpc_common::TimeInterval timer_l;
            LightSpmv(q, compute_units, work_group_size, row_counter_space, row, sum, n, nonzero, matrix, x, y_lightspmv);
            elapsed_l += timer_l.Elapsed();
            
            // Merge-based compute
            dpc_common::TimeInterval timer_m;
            MergeSparseMatrixVector(q, compute_units, work_group_size, n, nonzero, matrix, x, y_parallel, carry_row, carry_value);
            elapsed_m += timer_m.Elapsed();
        }
        
        elapsed_l /= repetitions;
        elapsed_m /= repetitions;
        
        double gflops_l = (double)(2 * nonzero - 1) / (elapsed_l * 1000000000);
        double gflops_m = (double)(2 * nonzero - 1) / (elapsed_m * 1000000000);
        
        std::cout << "nonzero: " << nonzero << ", numRows: " << n <<  ", mean: " << nonzero / n << ", LightSpMV: " << elapsed_l << " sec" << ", gflops: " << gflops_l <<" .\n";
        std::cout << "nonzero: " << nonzero << ", numRows: " << n <<  ", mean: " << nonzero / n << ", Merge-based: " << elapsed_m << " sec" << ", gflops: " << gflops_m <<" .\n";

        FreeMemory(q, &matrix, x, y_sequential, y_parallel, y_lightspmv, carry_row, carry_value, row_counter_space, row, sum);  
        } catch (std::exception const &e) {
            cout << "An exception is caught while computing on device.\n";
            terminate();
    }
    return;
}


int main(){
    /*------------------------------------------------------------------------------------------------------------------*/
    /*                                      show the device information                                                 */
    /*------------------------------------------------------------------------------------------------------------------*/
    queue q{default_selector_v};
    auto device = q.get_device();

    cout << "Device: " << device.get_info<info::device::name>() << "\n";
    cout << "local_mem_size: " << q.get_device().get_info<info::device::local_mem_size>() << "\n";

    int compute_units = 32;
    int work_group_size = 32;
    int thread_count = compute_units * work_group_size;

    cout << "Compute units: " << compute_units << "\n";
    cout << "Work group size: " << work_group_size << "\n";
      
    // get all supported sub_group sizes and print
    auto sg_sizes = device.get_info<info::device::sub_group_sizes>();
    std::cout << "Supported Sub-Group Sizes : ";
    for (int i=0; i<sg_sizes.size(); i++) std::cout << sg_sizes[i] << " "; std::cout << "\n";
    
    /*------------------------------------------------------------------------------------------------------------------*/
    /*                                            do the experiment                                                     */
    /*------------------------------------------------------------------------------------------------------------------*/
    
    // n x n sparse matrix.
    int n = 10 * 1000;
    
    // Number of non zero values in sparse matrix.
    int nonzero;
    
    // Number of repetitions.
    int repetitions = 50; 
    
    // Maximum value of an element in the matrix.
    constexpr int max_value = 100;
    
    // do the experment, find the best vector_size
    // for(nonzero = n; nonzero <= 32 * n; nonzero += n){
    //     exp_lightspmv(n, nonzero, repetitions, max_value, compute_units, work_group_size);
    // }
    
    /*------------------------------------------------------------------------------------------------------------------*/
    /*                                               check speed                                                        */
    /*------------------------------------------------------------------------------------------------------------------*/
    
    // do the experment, check the speed of alg
    for(nonzero = 1 * n; nonzero < 32 * n; nonzero += 1 * n){
        exp_all(n, nonzero, repetitions, max_value, compute_units, work_group_size);
    }
    
    
}