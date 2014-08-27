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
__global__ void setBoundryConditions_kernel(int nWidth, int nHeight, int nDepth, pycuda::complex<cudaP> *psi){
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
  int tid_b = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
//   int bid = blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y;
  
  pyComplex psi = psi_d[tid];
  __shared__ cudaP density[ %(THREADS_PER_BLOCK)s ];
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
__global__ void getVelocity_kernel( int neighbors, cudaP dx, cudaP dy, cudaP dz, pyComplex *psi, unsigned char *activity, cudaP *psiOther){
  int t_j = blockIdx.x*blockDim.x + threadIdx.x;
  int t_i = blockIdx.y*blockDim.y + threadIdx.y;
  int t_k = blockIdx.z*blockDim.z + threadIdx.z;
  int tid = t_j + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
  int tid_b = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
  int bid = blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y;
  
  //Border blocks are skiped
  if ( ( blockIdx.x == 0 or blockDim.x == gridDim.x -1 ) or ( blockIdx.y == 0 or blockDim.y == gridDim.y -1 ) or ( blockIdx.z == 0 or blockDim.z == gridDim.z -1 ) ) return; 
 
  __shared__ unsigned char activeBlock;
  if (tid_b == 0 ) activeBlock = activity[bid];
  __syncthreads();
  if ( !activeBlock ) return; 
  pyComplex center = psi[tid];
  __shared__ pyComplex psi_sh[ %(B_WIDTH)s ][ %(B_HEIGHT)s ][ %(B_DEPTH)s ];
  __syncthreads();
  psi_sh[threadIdx.x][threadIdx.y][threadIdx.z] = center;
  __syncthreads();
    
  cudaP dxInv = cudaP(1.0)/dx;
  cudaP dyInv = cudaP(1.0)/dy;
  cudaP dzInv = cudaP(1.0)/dz;
  pyComplex gradient_x, gradient_y, gradient_z;
    

  if ( threadIdx.x == 0 ) gradient_x = ( psi_sh[threadIdx.x+1][threadIdx.y][threadIdx.z] - center )*dxInv;
  else if ( threadIdx.x == (blockDim.x-1) ) gradient_x = ( center - psi_sh[threadIdx.x-1][threadIdx.y][threadIdx.z] )*dxInv;
  else gradient_x = ( psi_sh[threadIdx.x+1][threadIdx.y][threadIdx.z] - psi_sh[threadIdx.x-1][threadIdx.y][threadIdx.z] ) * dxInv* cudaP(0.5);

  if ( threadIdx.y == 0 ) gradient_y = ( psi_sh[threadIdx.x][threadIdx.y+1][threadIdx.z] - center )*dyInv;
  else if ( threadIdx.y == (blockDim.y-1) ) gradient_y = ( center - psi_sh[threadIdx.x][threadIdx.y-1][threadIdx.z] )*dyInv;
  else gradient_y = ( psi_sh[threadIdx.x][threadIdx.y+1][threadIdx.z] - psi_sh[threadIdx.x][threadIdx.y-1][threadIdx.z] ) * dyInv* cudaP(0.5);
  
  if ( threadIdx.z == 0 ) gradient_z = ( psi_sh[threadIdx.x][threadIdx.y][threadIdx.z+1] - center )*dzInv;
  else if ( threadIdx.z == (blockDim.z-1) ) gradient_z = ( center - psi_sh[threadIdx.x][threadIdx.y][threadIdx.z-1] )*dzInv;
  else gradient_z = ( psi_sh[threadIdx.x][threadIdx.y][threadIdx.z+1] - psi_sh[threadIdx.x][threadIdx.y][threadIdx.z-1] ) * dzInv* cudaP(0.5);

  __syncthreads();
  cudaP rho = norm(center) + cudaP(0.000005);
  cudaP velX = (center._M_re*gradient_x._M_im - center._M_im*gradient_x._M_re)/rho;
  cudaP velY = (center._M_re*gradient_y._M_im - center._M_im*gradient_y._M_re)/rho;
  cudaP velZ = (center._M_re*gradient_z._M_im - center._M_im*gradient_z._M_re)/rho; 

  psiOther[tid] =  sqrt( velX*velX + velY*velY + velZ*velZ ) ;
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__device__ pyComplex vortexCore( int nWidth, int nHeight, int nDepth, cudaP xMin, cudaP yMin, cudaP zMin, 
				 cudaP dx, cudaP dy, cudaP dz, int t_i, int t_j, int t_k, int tid,
				 cudaP gammaX, cudaP gammaY, cudaP gammaZ, cudaP omega, cudaP x0, cudaP y0, pyComplex *psiStep){
  pyComplex center = psiStep[tid];
  __shared__ pyComplex psi_sh[ %(B_WIDTH)s ][ %(B_HEIGHT)s ][ %(B_DEPTH)s ];
  __syncthreads();
  psi_sh[threadIdx.x][threadIdx.y][threadIdx.z] = center;
  __syncthreads();
  
  cudaP dxInv = cudaP(1.0)/dx;
  cudaP dyInv = cudaP(1.0)/dy;
  cudaP dzInv = cudaP(1.0)/dz;
  cudaP x = t_j*dx + xMin;
  cudaP y = t_i*dy + yMin;
  cudaP z = t_k*dz + zMin;
  
  pyComplex iComplex( cudaP(0.), cudaP(1.) );
  pyComplex laplacian( cudaP(0.), cudaP(0.) );
  pyComplex Lz( cudaP(0.), cudaP(0.) );
  pyComplex psiMinus, psiPlus;
  
  // laplacian X-term
  if (threadIdx.x==0 ){
    psiMinus = psiStep[ (t_j-1) + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y ]; 
    psiPlus  = psi_sh[threadIdx.x+1][threadIdx.y][threadIdx.z];
  }
  else if (threadIdx.x==blockDim.x-1){
    psiPlus  = psiStep[ (t_j+1) + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y ];
    psiMinus = psi_sh[threadIdx.x-1][threadIdx.y][threadIdx.z];    
  }
  else {
    psiPlus  = psi_sh[threadIdx.x+1][threadIdx.y][threadIdx.z];
    psiMinus = psi_sh[threadIdx.x-1][threadIdx.y][threadIdx.z];
  }
//   psiMinus = threadIdx.x == 0 ? 
// 	      psiStep[ (t_j-1) + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y ] : 
// 	      psi_sh[threadIdx.x-1][threadIdx.y][threadIdx.z];
//   psiPlus  = threadIdx.x == (nWidth-1) ? 
// 	      psiStep[ (t_j+1) + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y ] : 
// 	      psi_sh[threadIdx.x+1][threadIdx.y][threadIdx.z];
  laplacian += ( psiPlus + psiMinus - cudaP(2.)*center )*dxInv*dxInv;
  Lz += -iComplex*( psiPlus - psiMinus )*dxInv*cudaP(0.5)*(y-y0)*omega;
  
  // laplacian Y-term
  if (threadIdx.y==0 ){
    psiMinus = psiStep[ t_j + (t_i-1)*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y ]; 
    psiPlus  = psi_sh[threadIdx.x][threadIdx.y+1][threadIdx.z];
  }
  else if (threadIdx.y==blockDim.y-1){
    psiPlus  = psiStep[ t_j + (t_i+1)*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y ];
    psiMinus = psi_sh[threadIdx.x][threadIdx.y-1][threadIdx.z];    
  }
  else {
    psiPlus  = psi_sh[threadIdx.x][threadIdx.y+1][threadIdx.z];
    psiMinus = psi_sh[threadIdx.x][threadIdx.y-1][threadIdx.z];
  }
  laplacian += ( psiPlus + psiMinus - cudaP(2.)*center )*dyInv*dyInv;
  Lz +=  iComplex*( psiPlus - psiMinus)*dyInv*cudaP(0.5)*(x-x0)*omega;
  
  // laplacian Z-term
  if (threadIdx.z==0 ){
    psiMinus = psiStep[ t_j + t_i*blockDim.x*gridDim.x + (t_k-1)*blockDim.x*gridDim.x*blockDim.y*gridDim.y ]; 
    psiPlus  = psi_sh[threadIdx.x][threadIdx.y][threadIdx.z+1];
  }
  else if (threadIdx.z==blockDim.z-1){
    psiPlus  = psiStep[ t_j + t_i*blockDim.x*gridDim.x + (t_k+1)*blockDim.x*gridDim.x*blockDim.y*gridDim.y ];
    psiMinus = psi_sh[threadIdx.x][threadIdx.y][threadIdx.z-1];    
  }
  else {
    psiPlus  = psi_sh[threadIdx.x][threadIdx.y][threadIdx.z+1];
    psiMinus = psi_sh[threadIdx.x][threadIdx.y][threadIdx.z-1];
  }
  laplacian += ( psiPlus + psiMinus - cudaP(2.)*center )*dzInv*dzInv; 
  
  
  pyComplex Vtrap, GP; 
  Vtrap = (gammaX*x*x + gammaY*y*y + gammaZ*z*z)*cudaP(0.5)*center;
  GP = cudaP(8000.)*norm(center)*center;  
  
  return iComplex*(laplacian*cudaP(0.5) - Vtrap - GP - Lz);

}
////////////////////////////////////////////////////////////////////////////////
//////////////////////           EULER                //////////////////////////
////////////////////////////////////////////////////////////////////////////////
__global__ void eulerStep_kernel( int nWidth, int nHeight, int nDepth, cudaP slopeCoef, cudaP weight, 
				      cudaP xMin, cudaP yMin, cudaP zMin, cudaP dx, cudaP dy, cudaP dz, cudaP dt, 
				      cudaP gammaX, cudaP gammaY, cudaP gammaZ, cudaP x0, cudaP y0, cudaP omega,
				      pyComplex *psi_d, pyComplex *psiStepIn, pyComplex *psiStepOut, pyComplex *psiRunge,
				      unsigned char lastRK4Step, unsigned char *activity ){
  int t_j = blockIdx.x*blockDim.x + threadIdx.x;
  int t_i = blockIdx.y*blockDim.y + threadIdx.y;
  int t_k = blockIdx.z*blockDim.z + threadIdx.z;
  int tid = t_j + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
  int tid_b = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
//   int bid = blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y;
  
  
  //Border blocks are skiped
  if ( ( blockIdx.x == 0 or blockDim.x == gridDim.x -1 ) or ( blockIdx.y == 0 or blockDim.y == gridDim.y -1 ) or ( blockIdx.z == 0 or blockDim.z == gridDim.z -1 ) ) return; 
  //Unactive blocks are skiped
  __shared__ unsigned char activeBlock;
  if (tid_b == 0 ) activeBlock = activity[blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y];
  __syncthreads();
  if ( !activeBlock ) return;
  
  pyComplex value;
  value = vortexCore( nWidth, nHeight, nDepth, xMin, yMin, zMin, 
		      dx, dy, dz, t_i, t_j, t_k, tid, 
		      gammaX, gammaY, gammaZ, omega, x0, y0, psiStepIn );
  value = dt*value;
  
  if (lastRK4Step ){
    pyComplex valueOut = psiRunge[tid] + slopeCoef*value/cudaP(6.); 
    psiRunge[tid] = valueOut;
    psiStepOut[tid] = valueOut;
    psi_d[tid] = valueOut;
  }  
  else{
    psiStepOut[tid] = psi_d[tid] + weight*value;
    //add to rk4 final value
    psiRunge[tid] = psiRunge[tid] + slopeCoef*value/cudaP(6.);
  }
}

