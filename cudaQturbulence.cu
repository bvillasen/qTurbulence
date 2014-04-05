#include <pycuda-complex.hpp>
#define pi 3.14159265f
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// __device__ cudaP gValue(float t, float x, float y, float z, float constG, float ampG, float omegaG ){
// //   return constG + ampG*sin(2*3.14159f*omegaG*t)*cos(2*3.14159f*x/30)*cos(2*3.14159f*x/30);
// //   return constG ;//+ ampG*sin(2*3.14159f*omegaG*t);
//   return 8000.0f;
// }
// 
// __device__ float MfieldBX( float x, float y, float z, float t, float constMag, float ampMag, float omegaMag ){
// //   return 6000.0f*cos(2*pi*x/6.0f)*cos(2*pi*x/6.0f) + 2000;
//   return x ;//+2*sin(pi*t);
// //   return -cos(0.8*y)*sin(0.8*x);
//   
// }
// __device__ float MfieldBY( float x, float y, float z, float t, float constMag, float ampMag, float omegaMag ){
// //   return 6000.0f*cos(2*pi*x/6.0f)*cos(2*pi*x/6.0f) + 2000;
//   return -(y - constMag +ampMag*sin(t*omegaMag));
// //   return cos(0.8*x)*sin(0.8*y);
// }
// __device__ float MfieldBZ( float x, float y, float z, float t ){
// //   return 6000.0f*cos(2*pi*x/6.0f)*cos(2*pi*x/6.0f) + 2000;
//   return 0;
// }
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void getAlphas_kernel( cudaP dx, cudaP dy, cudaP dz, cudaP xMin, cudaP yMin, cudaP zMin,
				  cudaP gammaX, cudaP gammaY, cudaP gammaZ, 
				  pycuda::complex<cudaP> *psi1, cudaP *alphas){
  int t_j = blockIdx.x*blockDim.x + threadIdx.x;
  int t_i = blockIdx.y*blockDim.y + threadIdx.y;
  int t_k = blockIdx.z*blockDim.z + threadIdx.z;
  int tid = t_j + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
  
  cudaP x = t_j*dx + xMin;
  cudaP y = t_i*dy + yMin;
  cudaP z = t_k*dz + zMin;
  
  cudaP g = 8000;
  cudaP psi_mod = abs(psi1[tid]);
  cudaP result = (cudaP(0.5)*( gammaX*x*x + gammaY*y*y + gammaZ*z*z )) + g*psi_mod*psi_mod;
  alphas[tid] = result;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void solvePartialX_kernel( cudaP Lx, pycuda::complex<cudaP> *fftTrnf, pycuda::complex<cudaP> *partialxfft, cudaP *kxfft){
  int t_j = blockIdx.x*blockDim.x + threadIdx.x;
  int t_i = blockIdx.y*blockDim.y + threadIdx.y;
  int t_k = blockIdx.z*blockDim.z + threadIdx.z;
  int tid = t_j + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;

  cudaP kx = kxfft[t_j];
  pycuda::complex<cudaP> i_complex( cudaP(0.), cudaP(1.0));
  
  partialxfft[tid] = kx*i_complex*fftTrnf[tid];
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void solvePartialY_kernel( cudaP Ly, pycuda::complex<cudaP> *fftTrnf, pycuda::complex<cudaP> *partialyfft, cudaP *kyfft){
  int t_j = blockIdx.x*blockDim.x + threadIdx.x;
  int t_i = blockIdx.y*blockDim.y + threadIdx.y;
  int t_k = blockIdx.z*blockDim.z + threadIdx.z;
  int tid = t_j + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;

  cudaP ky = kyfft[t_i];
  pycuda::complex<cudaP> i_complex( cudaP(0.), cudaP(1.0));
  
  partialyfft[tid] = ky*i_complex*fftTrnf[tid];
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void implicitStep1_kernel( cudaP xMin, cudaP yMin, cudaP zMin, cudaP dx, cudaP dy, cudaP dz,  
			     cudaP alpha, cudaP omega, cudaP gammaX, cudaP gammaY, cudaP gammaZ, 
			     pycuda::complex<cudaP> *partialX_d,
			     pycuda::complex<cudaP> *partialY_d,
			     pycuda::complex<cudaP> *psi1_d, pycuda::complex<cudaP> *G1_d,
			     cudaP x0,cudaP y0){
  int t_j = blockIdx.x*blockDim.x + threadIdx.x;
  int t_i = blockIdx.y*blockDim.y + threadIdx.y;
  int t_k = blockIdx.z*blockDim.z + threadIdx.z;
  int tid = t_j + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
  
  cudaP x = t_j*dx + xMin;
  cudaP y = t_i*dy + yMin;
  cudaP z = t_k*dz + zMin;
  
  pycuda::complex<cudaP> iComplex( cudaP(0.0), cudaP(1.0) );
  pycuda::complex<cudaP> complex1( cudaP(1.0), cudaP(0.0) );
  pycuda::complex<cudaP> psi1, partialX, partialY, Vtrap, torque, lz, result;
  cudaP g = 8000;
  
  psi1 = psi1_d[tid];
  cudaP psiMod = abs(psi1);
  Vtrap = psi1*(gammaX*x*x + gammaY*y*y + gammaZ*z*z)*cudaP(0.5);
  torque = psi1*g*psiMod*psiMod;
  partialX = partialX_d[tid];
  partialY = partialY_d[tid];
  lz = iComplex * omega * (partialY*(x-x0) - partialX*(y-y0)); 
  G1_d[tid] = psi1*alpha - Vtrap - torque - lz;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void implicitStep2_kernel( cudaP dt, cudaP *kx, cudaP *ky, cudaP *kz, cudaP alpha, 
			      pycuda::complex<cudaP> *psiTransf, pycuda::complex<cudaP> *GTranf){
  int t_j = blockIdx.x*blockDim.x + threadIdx.x;
  int t_i = blockIdx.y*blockDim.y + threadIdx.y;
  int t_k = blockIdx.z*blockDim.z + threadIdx.z;
  int tid = t_j + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
  
  cudaP kX = kx[t_j];
  cudaP kY = ky[t_i];
  cudaP kZ = kz[t_k];
  cudaP k2 = kX*kX + kY*kY + kZ*kZ;
  
  pycuda::complex<cudaP> factor, timeStep, psiT, Gt;
  factor = cudaP(2.0) / ( 2 + dt*(k2 + 2*alpha) );
  psiT = psiTransf[tid];
  Gt = GTranf[tid];
  psiTransf[tid] = factor * ( psiT + Gt*dt); 
  
}
