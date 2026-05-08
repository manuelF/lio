#ifndef __G2G_MATRIX_H__
#define __G2G_MATRIX_H__

#include <vector>
#include <cassert>

// TODO: para cuando es multiplo, le suma igual y no deberia
#define COALESCED_DIMENSION(d) (d + 32 - (d % 32))

#include "cuda_includes.h"
#include "scalar_vector_types.h"

namespace G2G {
  enum UpperLowerTriangle { UpperTriangle, LowerTriangle };

  template<class T> class Matrix {
    public:
      Matrix(void) noexcept;
      virtual ~Matrix(void) noexcept;

      T* data;
      unsigned int width;
      unsigned int height;

      unsigned int bytes(void) const;
      unsigned int elements(void) const;

      bool is_allocated(void) const;

      virtual void deallocate(void) noexcept = 0;
  };

  template<class T> class CudaMatrix;

  template<class T> class HostMatrix : public Matrix<T> {
    public:
      enum PinnedFlag { Pinned, NonPinned };

      HostMatrix(PinnedFlag pinned = NonPinned) noexcept;
      HostMatrix(unsigned int width, unsigned int height = 1, PinnedFlag pinned = NonPinned) noexcept;
      HostMatrix(const CudaMatrix<T>& c) noexcept;
      HostMatrix(const HostMatrix<T>& c) noexcept;
      ~HostMatrix(void) noexcept;

      HostMatrix<T>& operator=(const CudaMatrix<T>& c) noexcept;
      HostMatrix<T>& operator=(const HostMatrix<T>& c) noexcept;

      inline const T& operator()(unsigned int i = 0, unsigned int j = 0) const {
        assert(i < this->width);
        assert(j < this->height);
        return this->data[j * stride + i];
      }

      inline T& operator()(unsigned int i = 0, unsigned int j = 0) {
        assert(i < this->width);
        assert(j < this->height);
        return this->data[j * stride + i];
      }

      inline const T * row(unsigned int i = 0) const {
        return &this->data[i * stride];
      }

      inline const T * asArray() const {
        return this->data;
      }

      inline T* ptr(unsigned int i = 0, unsigned int j = 0) {
        assert(i < this->width);
        assert(j < this->height);
        return &this->data[j * stride + i];
      }

      // Total allocated bytes (stride * height, accounts for row padding).
      inline unsigned int alloc_bytes() const {
        return stride * this->height * sizeof(T);
      }

      void copy_to_tmp(T *) const noexcept;
      void copy_submatrix(const CudaMatrix<T>& c, unsigned int elements = 0) noexcept;
#if GPU_KERNELS
      void copy_submatrix_async(const CudaMatrix<T>& c, cudaStream_t stream, unsigned int elements = 0) noexcept;
#endif
      void copy_submatrix(const HostMatrix<T>& c, unsigned int elements = 0) noexcept;

      void copy_transpose(const CudaMatrix<T>& cuda_matrix) noexcept;
      void transpose(HostMatrix<T>& out) const noexcept;

      HostMatrix<T>& resize(unsigned int width, unsigned int height = 1) noexcept;
      HostMatrix<T>& shrink(unsigned int width, unsigned int height = 1) noexcept;
      HostMatrix<T>& zero(void) noexcept;
      HostMatrix<T>& fill(T value) noexcept;

      void deallocate(void) noexcept;

      void to_constant(const char* constant);

      // Row stride in elements (>= width).  Padded to 64-byte boundary so
      // that every row(j) pointer is 64-byte aligned when the base is.
      unsigned int stride;

    private:
      bool pinned;
      void alloc_data(void) noexcept;
      void dealloc_data(void) noexcept;

      // Compute stride: round width up to the next multiple of 64/sizeof(T)
      // so that consecutive rows are cache-line aligned.
      static unsigned int compute_stride(unsigned int w) {
        const unsigned int elems_per_line = 64 / sizeof(T);
        if (elems_per_line <= 1) return w;
        return (w + elems_per_line - 1) / elems_per_line * elems_per_line;
      }
  };

  template<class T> void to_constant(const char* constant, const T& value);

  template <class T> class CudaMatrix : public Matrix<T> {
    public:
      CudaMatrix(void) noexcept;
      CudaMatrix(const CudaMatrix<T>& c) noexcept;
      CudaMatrix(const HostMatrix<T>& c) noexcept;
      CudaMatrix(const std::vector<T>& v) noexcept;

      CudaMatrix(unsigned int width, unsigned int height = 1) noexcept;
      ~CudaMatrix(void) noexcept;

      CudaMatrix<T>& resize(unsigned int width, unsigned int height = 1) noexcept;
      CudaMatrix<T>& zero(void) noexcept;

      CudaMatrix& operator=(const HostMatrix<T>& c) noexcept;
      CudaMatrix& operator=(const CudaMatrix<T>& c) noexcept;
      CudaMatrix& operator=(const std::vector<T>& v) noexcept;

      void copy_submatrix(const HostMatrix<T>& c, unsigned int elements = 0) noexcept;
      void copy_submatrix(const CudaMatrix<T>& c, unsigned int elements = 0) noexcept;
      void copy_submatrix(const std::vector<T>& v, unsigned int elements = 0) noexcept;

      void deallocate(void) noexcept;
  };

  template <class T> class FortranMatrix : public Matrix<T> {
    public:
      FortranMatrix(void) noexcept;
      FortranMatrix(T* ptr, unsigned int width, unsigned int height = 1, unsigned int fortran_width = 1) noexcept;

      inline T& operator()(unsigned int x = 0, unsigned int y = 0) {
        assert(x < this->width);
        assert(y < this->height);
        return this->data[y * fortran_width + x];
      }
      inline const T& operator()(unsigned int x = 0, unsigned int y = 0) const {
        assert(x < this->width);
        assert(y < this->height);
        return this->data[y * fortran_width + x];
      }
      void deallocate(void) noexcept {};

    private:
      unsigned int fortran_width;
  };

  typedef HostMatrix<double> HostMatrixDouble;
  typedef HostMatrix<double3> HostMatrixDouble3;
  typedef HostMatrix<float> HostMatrixFloat;
  typedef HostMatrix<float1> HostMatrixFloat1;
  typedef HostMatrix<float3> HostMatrixFloat3;
  typedef HostMatrix<float4> HostMatrixFloat4;
  typedef HostMatrix<uint> HostMatrixUInt;
  typedef HostMatrix<uint1> HostMatrixUInt1;
  typedef HostMatrix<uint2> HostMatrixUInt2;

  typedef CudaMatrix<float> CudaMatrixFloat;
  typedef CudaMatrix<float1> CudaMatrixFloat1;
  typedef CudaMatrix<float2> CudaMatrixFloat2;
  typedef CudaMatrix<float3> CudaMatrixFloat3;
  typedef CudaMatrix<float4> CudaMatrixFloat4;
  typedef CudaMatrix<uint> CudaMatrixUInt;
  typedef CudaMatrix<uint1> CudaMatrixUInt1;
  typedef CudaMatrix<uint2> CudaMatrixUInt2;
}

#endif
