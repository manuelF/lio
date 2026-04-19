#include <iostream>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include "common.h"
#include "matrix.h"
#include "scalar_vector_types.h"
using namespace std;

namespace G2G {

/***************************
 * Matrix
 ***************************/

template<class T> Matrix<T>::Matrix(void) noexcept : data(NULL), width(0), height(0) /*, components(0)*/ {}

template<class T> Matrix<T>::~Matrix(void) noexcept { }

template<class T> unsigned int Matrix<T>::bytes(void) const {
	return elements() * sizeof(T);
}

template<class T> unsigned int Matrix<T>::elements(void) const {
	return width * height /* * components */;
}

template<class T> bool Matrix<T>::is_allocated(void) const {
	return data;
}

/***************************
 * HostMatrix
 ***************************/
template<class T> void HostMatrix<T>::alloc_data(void) noexcept {
  unsigned int nbytes = alloc_bytes();
  assert(nbytes != 0);
  int posix_return = 0;

  void* raw_ptr = nullptr;
  if (pinned) {
#if GPU_KERNELS
    cudaError_t error_status = cudaMallocHost(&raw_ptr, nbytes);
    assert(error_status == cudaSuccess);
#else
    assert(false);
#endif
  }
  else
  {
    posix_return = posix_memalign(&raw_ptr, 64, nbytes);
    if ( posix_return != 0)
    {
       std::cout <<"HostMatrix: Error in posix_memalign.\n";
       exit(1);
    };
  };
  this->data = static_cast<T*>(raw_ptr);

  assert(this->data);
}

template<class T> void HostMatrix<T>::dealloc_data(void) noexcept {
	if (pinned) {
    #if GPU_KERNELS
    cudaFreeHost(this->data);
    #else
    assert(false);
    #endif
  }
	else free(this->data); //mkl_free(this->data);
}

template<class T> void HostMatrix<T>::copy_to_tmp(T * dst) const noexcept {
    memcpy(dst, this->data, this->alloc_bytes());
}

template<class T> void HostMatrix<T>::deallocate(void) noexcept {
	dealloc_data();
	this->data = NULL;
  this->width = this->height = 0;
  stride = 0;
}

template<class T> HostMatrix<T>::HostMatrix(PinnedFlag _pinned) noexcept : Matrix<T>() {
  pinned = (_pinned == Pinned);
  stride = 0;
}

template<class T> HostMatrix<T>::HostMatrix(unsigned int _width, unsigned _height, PinnedFlag _pinned) noexcept : Matrix<T>() {
  pinned = (_pinned == Pinned);
  stride = 0;
  resize(_width, _height);
}

template<class T> HostMatrix<T>::HostMatrix(const CudaMatrix<T>& c) noexcept : Matrix<T>(), stride(0), pinned(false) {
	*this = c;
}

template<class T> HostMatrix<T>::HostMatrix(const HostMatrix<T>& m) noexcept : Matrix<T>(), stride(0), pinned(m.pinned) {
	if (m.data) {
		this->width = m.width; this->height = m.height;
		stride = compute_stride(this->width);
		alloc_data();
		copy_submatrix(m);
	}
}

template<class T> HostMatrix<T>::~HostMatrix(void) noexcept {
	deallocate();
}

template<class T> HostMatrix<T>& HostMatrix<T>::resize(unsigned int _width, unsigned _height) noexcept {
  assert(_width != 0 && "HostMatrix::resize: width cannot be 0");
  assert(_height != 0 && "HostMatrix::resize: height cannot be 0");
  if (_width != this->width || _height != this->height) {
    if (this->data) dealloc_data();
    this->width = _width; this->height = _height;
    stride = compute_stride(_width);
    alloc_data();
  }

	return *this;
}

template<class T> HostMatrix<T>& HostMatrix<T>::shrink(unsigned int _width, unsigned int _height) noexcept {
  assert((_width != 0 && _height != 0) && "HostMatrix::shrink: dimensions cannot be 0");
  if (_width != this->width || _height != this->height) {
    HostMatrix<T> temp_matrix(_width, _height);
    temp_matrix.copy_submatrix(temp_matrix, _width * _height);
    resize(_width, _height);
    *this = temp_matrix;
  }

	return *this;
}

template<class T> HostMatrix<T>& HostMatrix<T>::zero(void) noexcept {
  std::fill(this->data, this->data + stride * this->height, T{});
	return *this;
}

template<class T> HostMatrix<T>& HostMatrix<T>::fill(T value) noexcept {
  unsigned int total = stride * this->height;
  for (uint i = 0; i < total; i++) { this->data[i] = value; }
  return *this;
}

template<class T> HostMatrix<T>& HostMatrix<T>::operator=(const HostMatrix<T>& c) noexcept {

	if (!c.data) {
		if (this->data) { dealloc_data(); this->width = this->height = 0; stride = 0; this->data = NULL; }
	}
	else {
		if (this->data) {
			if (this->width != c.width || this->height != c.height) {
				dealloc_data();
				this->width = c.width; this->height = c.height;
				stride = compute_stride(this->width);
				alloc_data();
			}
		}
		else {
			this->width = c.width; this->height = c.height;
			stride = compute_stride(this->width);
			alloc_data();
		}

		copy_submatrix(c);
	}

	return *this;
}

template <class T> HostMatrix<T>& HostMatrix<T>::operator=(const CudaMatrix<T>& c) noexcept {
	if (!c.data) {
		if (this->data) { dealloc_data(); this->width = this->height = 0; stride = 0; this->data = NULL; }
	}
	else {
		if (this->data) {
			if (this->width != c.width || this->height != c.height) {
				dealloc_data();
				this->width = c.width; this->height = c.height;
				stride = compute_stride(this->width);
				alloc_data();
			}
		}
		else {
			this->width = c.width; this->height = c.height;
			stride = compute_stride(this->width);
			alloc_data();
		}

		copy_submatrix(c);
	}

	return *this;
}

template<class T> void HostMatrix<T>::copy_submatrix(const HostMatrix<T>& c, unsigned int _elements) noexcept {
	unsigned int _bytes = (_elements == 0 ? this->alloc_bytes() : _elements * sizeof(T));
	assert(_bytes <= c.alloc_bytes() && "HostMatrix::copy_submatrix: source too small");
	memcpy(this->data, c.data, _bytes);
}

template<class T> void HostMatrix<T>::copy_submatrix(const CudaMatrix<T>& c, unsigned int _elements) noexcept {
	unsigned int _bytes = (_elements == 0 ? this->bytes() : _elements * sizeof(T));
	//cout << "bytes: " << _bytes << ", c.bytes: " << c.bytes() << endl;
	assert(_bytes <= c.bytes() && "HostMatrix::copy_submatrix: source too small");

  #if GPU_KERNELS
	cudaMemcpy(this->data, c.data, _bytes, cudaMemcpyDeviceToHost);
  cudaAssertNoError("HostMatrix::copy_submatrix");
  #else
  assert(false);
  #endif
}

#if GPU_KERNELS
template<class T> void HostMatrix<T>::copy_submatrix_async(const CudaMatrix<T>& c, cudaStream_t stream, unsigned int _elements) noexcept {
	unsigned int _bytes = (_elements == 0 ? this->bytes() : _elements * sizeof(T));
	assert(_bytes <= c.bytes() && "HostMatrix::copy_submatrix_async: source too small");

	cudaMemcpyAsync(this->data, c.data, _bytes, cudaMemcpyDeviceToHost, stream);
}
#endif

template<class T> void HostMatrix<T>::to_constant(const char* symbol) {
  #if GPU_KERNELS
	cudaMemcpyToSymbol(symbol, this->data, this->bytes(), 0, cudaMemcpyHostToDevice);
  cudaAssertNoError("to_constant");
  #endif
}

template<class T> void HostMatrix<T>::transpose(HostMatrix<T>& out) const noexcept {
  out.resize(this->height, this->width);
  out.zero();
  for (uint i = 0; i < this->width; i++) {
    for (uint j = 0; j < this->height; j++) {
      out(j, i) = (*this)(i, j);
    }
  }
}
template<class T> void HostMatrix<T>::copy_transpose(const CudaMatrix<T>& cuda_matrix) noexcept {
  assert((cuda_matrix.width == this->height && cuda_matrix.height == this->width) && "HostMatrix::copy_transpose: dimension mismatch");
  HostMatrix<T> cuda_matrix_copy(cuda_matrix);
  for (uint i = 0; i < cuda_matrix.width; i++) {
    for (uint j = 0; j < cuda_matrix.height; j++) {
      (*this)(j, i) = cuda_matrix_copy(i, j);
    }
  }
}

template<class T> void to_constant(const char* constant, const T& value) {
  #if GPU_KERNELS
	cudaMemcpyToSymbol(constant, &value, sizeof(T), 0, cudaMemcpyHostToDevice);
  cudaAssertNoError("to_constant(value)");
  #endif
}

template void to_constant<uint>(const char* constant, const uint& value);
template void to_constant<float>(const char* constant, const float& value);
template void to_constant<double>(const char* constant, const double& value);

/******************************
 * CudaMatrix
 ******************************/

template<class T> CudaMatrix<T>::CudaMatrix(void) noexcept : Matrix<T>() { }

template<class T> CudaMatrix<T>::CudaMatrix(unsigned int _width, unsigned int _height) noexcept : Matrix<T>() {
	resize(_width, _height);
}

template<class T> CudaMatrix<T>& CudaMatrix<T>::resize(unsigned int _width, unsigned int _height) noexcept {
  assert(_width * _height != 0);

  #if GPU_KERNELS
  if (_width != this->width || _height != this->height) {
    if (this->data) cudaFree(this->data);
    this->width = _width; this->height = _height;
    void* ptr = nullptr;
    cudaMalloc(&ptr, this->bytes());
    this->data = static_cast<T*>(ptr);
    cudaAssertNoError("CudaMatrix::resize");
  }
  #endif
	return *this;
}

template<class T> CudaMatrix<T>& CudaMatrix<T>::zero(void) noexcept {
  #if GPU_KERNELS
	assert(this->data);
	cudaMemset(this->data, 0, this->bytes());
  cudaAssertNoError("CudaMatrix::zero");
  #endif
	return *this;
}

template<class T> CudaMatrix<T>::CudaMatrix(const CudaMatrix<T>& c) noexcept : Matrix<T>() {
	*this = c;
}

template<class T> CudaMatrix<T>::CudaMatrix(const HostMatrix<T>& c) noexcept : Matrix<T>() {
	*this = c;
}

template<class T> CudaMatrix<T>::CudaMatrix(const std::vector<T>& v) noexcept : Matrix<T>() {
	*this = v;
}

template<class T> CudaMatrix<T>::~CudaMatrix(void) noexcept {
  deallocate();
}

template<class T> void CudaMatrix<T>::deallocate(void) noexcept {
  #if GPU_KERNELS
	if (this->data) cudaFree(this->data);
	this->data = NULL;
  this->width = this->height = 0;
  #endif
}

template<class T> void CudaMatrix<T>::copy_submatrix(const HostMatrix<T>& c, unsigned int _elements) noexcept {
	unsigned int _bytes = (_elements == 0 ? this->bytes() : _elements * sizeof(T));
	//cout << "bytes: " << _bytes << ", c.bytes: " << c.bytes() << endl;
	assert(_bytes <= c.bytes() && "CudaMatrix::copy_submatrix: source too small");

  #if GPU_KERNELS
	cudaMemcpy(this->data, c.data, _bytes, cudaMemcpyHostToDevice);
  cudaAssertNoError("CudaMatrix::copy_submatrix");
  #endif
}

template<class T> void CudaMatrix<T>::copy_submatrix(const CudaMatrix<T>& c, unsigned int _elements) noexcept {
	unsigned int _bytes = (_elements == 0 ? this->bytes() : _elements * sizeof(T));
	assert(_bytes <= c.bytes() && "CudaMatrix::copy_submatrix: source too small");

  #if GPU_KERNELS
	cudaMemcpy(c.data, this->data, _bytes, cudaMemcpyDeviceToDevice);
  cudaAssertNoError("CudaMatrix::copy_submatrix");
  #endif
}

template<class T> void CudaMatrix<T>::copy_submatrix(const std::vector<T>& v, unsigned int _elements) noexcept {
	unsigned int _bytes = (_elements == 0 ? this->bytes() : _elements * sizeof(T));
	assert(_bytes <= v.size() * sizeof(T) && "CudaMatrix::copy_submatrix: source too small");

  #if GPU_KERNELS
	cudaMemcpy(this->data, (T*)&v[0], _bytes, cudaMemcpyHostToDevice);
  cudaAssertNoError("CudaMatrix::copy_submatrix");
  #endif
}

template<class T> CudaMatrix<T>& CudaMatrix<T>::operator=(const HostMatrix<T>& c) noexcept {
  #if GPU_KERNELS
	if (!c.data) {
		if (this->data) { cudaFree(this->data); this->width = this->height = 0; this->data = NULL; }
	}
	else {
		void* ptr = nullptr;
		if (this->data) {
			if (this->bytes() != c.bytes()) {
				cudaFree(this->data);
				this->width = c.width; this->height = c.height;
				cudaMalloc(&ptr, this->bytes());
				this->data = static_cast<T*>(ptr);
			}
		}
		else {
			this->width = c.width; this->height = c.height;
			cudaMalloc(&ptr, this->bytes());
			this->data = static_cast<T*>(ptr);
		}
		cudaAssertNoError("CudaMatrix::operator=");
		copy_submatrix(c);
	}
  #endif
	return *this;
}

template<class T> CudaMatrix<T>& CudaMatrix<T>::operator=(const std::vector<T>& v) noexcept {
  #if GPU_KERNELS
	if (v.empty()) {
		if (this->data) { cudaFree(this->data); this->width = this->height = 0; this->data = NULL; }
	}
	else {
		void* ptr = nullptr;
		if (this->data) {
			if (this->elements() != v.size()) {
				cudaFree(this->data);
				this->width = static_cast<unsigned int>(v.size()); this->height = 1;
				cudaMalloc(&ptr, this->bytes());
				this->data = static_cast<T*>(ptr);
			}
		}
		else {
			this->width = static_cast<unsigned int>(v.size()); this->height = 1;
			cudaMalloc(&ptr, this->bytes());
			this->data = static_cast<T*>(ptr);
		}
    cudaAssertNoError("CudaMatrix::operator=");
		copy_submatrix(v);
	}
  #endif
	return *this;
}

template<class T> CudaMatrix<T>& CudaMatrix<T>::operator=(const CudaMatrix<T>& c) noexcept {
  #if GPU_KERNELS
	// copies data from c, only if necessary (always frees this's data, if any)
	if (!c.data) {
		if (this->data) { cudaFree(this->data); this->width = this->height = 0; this->data = NULL; }
	}
	else {
		void* ptr = nullptr;
		if (this->data) {
			if (this->bytes() != c.bytes()) {
				cudaFree(this->data);
				this->width = c.width; this->height = c.height;
				cudaMalloc(&ptr, this->bytes());
				this->data = static_cast<T*>(ptr);
			}
		}
		else {
			this->width = c.width; this->height = c.height;
			cudaMalloc(&ptr, this->bytes());
			this->data = static_cast<T*>(ptr);
    }
		cudaMemcpy(this->data, c.data, this->bytes(), cudaMemcpyDeviceToDevice);
	}
  cudaAssertNoError("CudaMatrix::operator=");
	#endif
	return *this;
}

/*************************************
 * FortranMatrix
 *************************************/
template<class T> FortranMatrix<T>::FortranMatrix(void) noexcept
	: Matrix<T>(), fortran_width(0)
{ }

template<class T> FortranMatrix<T>::FortranMatrix(T* _data, unsigned int _width, unsigned int _height, unsigned int _fortran_width) noexcept
	: Matrix<T>(), fortran_width(_fortran_width)
{
	this->data = _data;
	this->width = _width; this->height = _height;
	assert(this->data);
}

/**
 * Instantiations
 */
template class Matrix<double>;
template class Matrix<double3>;
template class Matrix<float>;
template class Matrix<float3>;
template class Matrix<uint>;

template class Matrix< vec_type<float, 2> >;
template class Matrix< vec_type<float, 3> >;
template class Matrix< vec_type<double, 2> >;
template class Matrix< vec_type<double, 3> >;

template class HostMatrix< vec_type<float, 2> >;
template class HostMatrix< vec_type<float, 3> >;
template class HostMatrix< vec_type<float, 4> >;
template class HostMatrix< vec_type<double, 2> >;
template class HostMatrix< vec_type<double, 3> >;
template class HostMatrix< vec_type<double, 4> >;
template class CudaMatrix< vec_type<float, 2> >;
template class CudaMatrix< vec_type<float, 3> >;
template class CudaMatrix< vec_type<float, 4> >;
template class CudaMatrix< vec_type<double, 2> >;
template class CudaMatrix< vec_type<double, 3> >;
template class CudaMatrix< vec_type<double, 4> >;

template class HostMatrix<double>;
template class HostMatrix<float>;

template class HostMatrix<double3>;
template class HostMatrix<float3>;
template class HostMatrix<uint>;

template class CudaMatrix<float>;
template class CudaMatrix<uint>;
template class CudaMatrix<double>;

template class FortranMatrix<double>;
template class FortranMatrix<unsigned int>;

}
