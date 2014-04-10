#include <pycuda-complex.hpp>
#define pi 3.14159265f
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
typedef   pycuda::complex<cudaP> pyComplex;
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void getAlphas_kernel( cudaP dx, cudaP dy, cudaP dz, cudaP xMin, cudaP yMin, cudaP zMin,
				  cudaP gammaX, cudaP gammaY, cudaP gammaZ, 
				  pyComplex *psi1, cudaP *alphas){
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
__global__ void solvePartialX_kernel( cudaP Lx, pyComplex *fftTrnf, pyComplex *partialxfft, cudaP *kxfft){
  int t_j = blockIdx.x*blockDim.x + threadIdx.x;
  int t_i = blockIdx.y*blockDim.y + threadIdx.y;
  int t_k = blockIdx.z*blockDim.z + threadIdx.z;
  int tid = t_j + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;

  cudaP kx = kxfft[t_j];
  pyComplex i_complex( cudaP(0.), cudaP(1.0));
  
  partialxfft[tid] = kx*i_complex*fftTrnf[tid];
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void solvePartialY_kernel( cudaP Ly, pyComplex *fftTrnf, pyComplex *partialyfft, cudaP *kyfft){
  int t_j = blockIdx.x*blockDim.x + threadIdx.x;
  int t_i = blockIdx.y*blockDim.y + threadIdx.y;
  int t_k = blockIdx.z*blockDim.z + threadIdx.z;
  int tid = t_j + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;

  cudaP ky = kyfft[t_i];
  pyComplex i_complex( cudaP(0.), cudaP(1.0));
  
  partialyfft[tid] = ky*i_complex*fftTrnf[tid];
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void setBoundryConditions_kernel(int nWidth, int nHeight, int nDepth, pycuda::complex<float> *psi){
  int t_j = blockIdx.x*blockDim.x + threadIdx.x;
  int t_i = blockIdx.y*blockDim.y + threadIdx.y;
  int t_k = blockIdx.z*blockDim.z + threadIdx.z;
  int tid = t_j + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
  
  if ( t_i==0 or t_i==(nHeight-1) or t_j==0 or t_j==(nWidth-1) or t_k==0 or t_k==(nDepth-1)){
    psi[tid] = 0;
  }
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void implicitStep1_kernel( cudaP xMin, cudaP yMin, cudaP zMin, cudaP dx, cudaP dy, cudaP dz,  
			     cudaP alpha, cudaP omega, cudaP gammaX, cudaP gammaY, cudaP gammaZ, 
			     pyComplex *partialX_d,
			     pyComplex *partialY_d,
			     pyComplex *psi1_d, pyComplex *G1_d,
			     cudaP x0,cudaP y0){
  int t_j = blockIdx.x*blockDim.x + threadIdx.x;
  int t_i = blockIdx.y*blockDim.y + threadIdx.y;
  int t_k = blockIdx.z*blockDim.z + threadIdx.z;
  int tid = t_j + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
  
  cudaP x = t_j*dx + xMin;
  cudaP y = t_i*dy + yMin;
  cudaP z = t_k*dz + zMin;
  
  pyComplex iComplex( cudaP(0.0), cudaP(1.0) );
  pyComplex complex1( cudaP(1.0), cudaP(0.0) );
  pyComplex psi1, partialX, partialY, Vtrap, torque, lz, result;
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
			      pyComplex *psiTransf, pyComplex *GTranf){
  int t_j = blockIdx.x*blockDim.x + threadIdx.x;
  int t_i = blockIdx.y*blockDim.y + threadIdx.y;
  int t_k = blockIdx.z*blockDim.z + threadIdx.z;
  int tid = t_j + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
  
  cudaP kX = kx[t_j];
  cudaP kY = ky[t_i];
  cudaP kZ = kz[t_k];
  cudaP k2 = kX*kX + kY*kY + kZ*kZ;
  
  pyComplex factor, timeStep, psiT, Gt;
  factor = cudaP(2.0) / ( 2 + dt*(k2 + 2*alpha) );
  psiT = psiTransf[tid];
  Gt = GTranf[tid];
  psiTransf[tid] = factor * ( psiT + Gt*dt); 
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void findActivity_kernel( cudaP minDensity, pyComplex *psi_d, unsigned char *activity ){
  int t_j = blockIdx.x*blockDim.x + threadIdx.x;
  int t_i = blockIdx.y*blockDim.y + threadIdx.y;
  int t_k = blockIdx.z*blockDim.z + threadIdx.z;
  int tid = t_j + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
  int tid_b = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.z;
//   int bid = blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y;
  
  pyComplex psi = psi_d[tid];
  __shared__ float density[ %(THREADS_PER_BLOCK)s ];
  density[tid_b] = abs(psi)*abs(psi);
  __syncthreads();
  
  int i = blockDim.x*blockDim.y*blockDim.z / 2;
  while ( i > 0 ){
    if ( tid_b < i ) density[tid_b] = density[tid_b] + density[tid_b+i];
    __syncthreads();
    i /= 2;
  }
  if ( tid_b == 0 ){
    if (density[0] >= minDensity ) {
      activity[ blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y ] = (unsigned char) 1;
      //right 
      if (blockIdx.x < gridDim.x-1) activity[ (blockIdx.x+1) + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y ] = (unsigned char) 1;
      //left
      if (blockIdx.x > 0) activity[ (blockIdx.x-1) + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y ] = (unsigned char) 1;
      //up 
      if (blockIdx.y < gridDim.y-1) activity[ blockIdx.x + (blockIdx.y+1)*gridDim.x + blockIdx.z*gridDim.x*gridDim.y ] = (unsigned char) 1;
      //down
      if (blockIdx.y > 0) activity[ blockIdx.x + (blockIdx.y-1)*gridDim.x + blockIdx.z*gridDim.x*gridDim.y ] = (unsigned char) 1;
      //top 
      if (blockIdx.z < gridDim.z-1) activity[ blockIdx.x + blockIdx.y*gridDim.x + (blockIdx.z+1)*gridDim.x*gridDim.y ] = (unsigned char) 1;
      //bottom
      if (blockIdx.z > 0) activity[ blockIdx.x + blockIdx.y*gridDim.x + (blockIdx.z-1)*gridDim.x*gridDim.y ] = (unsigned char) 1;
    }
  }
}
__global__ void getActivity_kernel( cudaP *psiOther, unsigned char *activity ){
  int t_j = blockIdx.x*blockDim.x + threadIdx.x;
  int t_i = blockIdx.y*blockDim.y + threadIdx.y;
  int t_k = blockIdx.z*blockDim.z + threadIdx.z;
  int tid = t_j + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
  int tid_b = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.z;
  int bid = blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y;
  
  __shared__ unsigned char activeBlock;
  if (tid_b == 0 ) activeBlock = activity[bid];
  __syncthreads();
  
  if ( activeBlock ) psiOther[tid] = cudaP(0.4);
  else psiOther[tid] = cudaP(0.08);
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void getVelocity_kernel( cudaP dx, cudaP dy, cudaP dz, pyComplex *psi, unsigned char *activity, cudaP *psiOther){
  int t_j = blockIdx.x*blockDim.x + threadIdx.x;
  int t_i = blockIdx.y*blockDim.y + threadIdx.y;
  int t_k = blockIdx.z*blockDim.z + threadIdx.z;
  int tid = t_j + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
  int tid_b = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.z;
  int bid = blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y;
  
  
  __shared__ unsigned char activeBlock;
  if (tid_b == 0 ) activeBlock = activity[bid];
  __syncthreads();
  if ( !activeBlock ) return; 
  pyComplex center = psi[tid];
  __shared__ pyComplex psi_sh[ %(B_WIDTH)s ][ %(B_HEIGHT)s ][ %(B_DEPTH)s ];
  psi_sh[threadIdx.x][threadIdx.y][threadIdx.z] = center;
  __syncthreads();
  
  
  
  
  cudaP dxInv = 1.0/dx;
  cudaP dyInv = 1.0/dy;
  cudaP dzInv = 1.0/dz;
  pyComplex gradient_x, gradient_y, gradient_z;
  
  
  
//   if ( ( blockIdx.x == 0 or blockDim.x == gridDim.x -1 ) or ( blockIdx.y == 0 or blockDim.y == gridDim.y -1 ) or ( blockIdx.z == 0 or blockDim.z == gridDim.z -1 ) ){ psiOther[tid] = cudaP(0.0); return; }
  
  if ( threadIdx.x == 0 ) gradient_x = ( psi_sh[threadIdx.x+1][threadIdx.y][threadIdx.z] - psi[ t_j-1 + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y ] )*dxInv*cudaP(0.5);
  else if ( threadIdx.x == blockDim.x-1 ) gradient_x = ( psi[ t_j+1 + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y ] - psi_sh[threadIdx.x-1][threadIdx.y][threadIdx.z] )*dxInv*cudaP(0.5);
  else gradient_x = ( psi_sh[threadIdx.x+1][threadIdx.y][threadIdx.z] - psi_sh[threadIdx.x-1][threadIdx.y][threadIdx.z] ) * dxInv* cudaP(0.5);

  if ( threadIdx.y == 0 ) gradient_y = ( psi_sh[threadIdx.x][threadIdx.y+1][threadIdx.z] - psi[ t_j + (t_i-1)*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y ] )*dyInv*cudaP(0.5);
  else if ( threadIdx.y == blockDim.y-1 ) gradient_y = ( psi[ t_j + (t_i+1)*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y ] - psi_sh[threadIdx.x][threadIdx.y-1][threadIdx.z] )*dyInv*cudaP(0.5);
  else gradient_y = ( psi_sh[threadIdx.x][threadIdx.y+1][threadIdx.z] - psi_sh[threadIdx.x][threadIdx.y-1][threadIdx.z] ) * dyInv* cudaP(0.5);
  
  if ( threadIdx.z == 0 ) gradient_z = ( psi_sh[threadIdx.x][threadIdx.y][threadIdx.z+1] - psi[ t_j + t_i*blockDim.x*gridDim.x + (t_k-1)*blockDim.x*gridDim.x*blockDim.y*gridDim.y ] )*dzInv*cudaP(0.5);
  else if ( threadIdx.z == blockDim.z-1 ) gradient_z = ( psi[ t_j + t_i*blockDim.x*gridDim.x + (t_k+1)*blockDim.x*gridDim.x*blockDim.y*gridDim.y ] - psi_sh[threadIdx.x][threadIdx.y][threadIdx.z-1] )*dzInv*cudaP(0.5);
  else gradient_z = ( psi_sh[threadIdx.x][threadIdx.y][threadIdx.z+1] - psi_sh[threadIdx.x][threadIdx.y][threadIdx.z-1] ) * dzInv* cudaP(0.5);
  
  cudaP rho = abs(center)*abs(center) + cudaP(0.000005);
  
//   if (multiplyBySqrtRho == 1){
//     velX[tid] = (center._M_re*gradient_x._M_im - center._M_im*gradient_x._M_re)/sqrt(rho);
//     velY[tid] = (center._M_re*gradient_y._M_im - center._M_im*gradient_y._M_re)/sqrt(rho);
//     velZ[tid] = (center._M_re*gradient_z._M_im - center._M_im*gradient_z._M_re)/sqrt(rho);
//   }
//   else{
  cudaP velX = (center._M_re*gradient_x._M_im - center._M_im*gradient_x._M_re)/rho;
  cudaP velY = (center._M_re*gradient_y._M_im - center._M_im*gradient_y._M_re)/rho;
  cudaP velZ = (center._M_re*gradient_z._M_im - center._M_im*gradient_z._M_re)/rho; 
//   
  psiOther[tid] = log( 1 + velX*velX + velY*velY + velZ*velZ ) ;
//   psiOther[tid] = rho; 
//   }
}
  


