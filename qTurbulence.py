import sys, time, os
import numpy as np
#import pylab as plt
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
#import pycuda.curandom as curandom
from pycuda.reduction import ReductionKernel

#Add Modules from other directories
currentDirectory = os.getcwd()
parentDirectory = currentDirectory[:currentDirectory.rfind("/")]
toolsDirectory = parentDirectory + "/tools"
volumeRenderDirectory = parentDirectory + "/volumeRender"
sys.path.extend( [toolsDirectory, volumeRenderDirectory] )

import volumeRender
from cudaTools import setCudaDevice, getFreeMemory, gpuArray3DtocudaArray


cudaP = "double"
nPoints = 128
useDevice = None
for option in sys.argv:
  if option == "float": cudaP = "float"
  if option == "128" or option == "256": nPoints = int(option)
  if option.find("device=") != -1: useDevice = int(option[-1]) 
 
precision  = {"float":(np.float32, np.complex64), "double":(np.float64,np.complex128) } 
cudaPre, cudaPreComplex = precision[cudaP]

#set simulation volume dimentions 
nWidth = nPoints
nHeight = nPoints
nDepth = nPoints
nData = nWidth*nHeight*nDepth

#Simulation Parameters
dt = 0.005

dtReal = 0.004
dtImag = 1e-10

Lx = 30.0
Ly = 30.0
Lz = 30.0
xMax, xMin = Lx/2, -Lx/2
yMax, yMin = Ly/2, -Ly/2
zMax, zMin = Lz/2, -Lz/2
dx, dy, dz = Lx/(nWidth-1), Ly/(nHeight-1), Lz/(nDepth-1 )
Z, Y, X = np.mgrid[ zMin:zMax:nDepth*1j, yMin:yMax:nHeight*1j, xMin:xMax:nWidth*1j ]

omega = 0.5
alpha = 1000.0
gammaX = 1.0
gammaY = 1.0
gammaZ = 1.0
x0 = cudaPre( 0. )
y0 = cudaPre( 0. )
neighbors = 1

plottingActive = False
plotVar = 0

#Change precision of the parameters
dx, dy, dz = cudaPre(dx), cudaPre(dy), cudaPre(dz)
Lx, Ly, Lz = cudaPre(Lx), cudaPre(Ly), cudaPre(Lz)
xMin, yMin, zMin = cudaPre(xMin), cudaPre(yMin), cudaPre(zMin)
dt = cudaPre(dt)
omega, gammaX, gammaY, gammaZ = cudaPre(omega), cudaPre(gammaX), cudaPre(gammaY), cudaPre(gammaZ), 
#Initialize openGL
volumeRender.nWidth = nWidth
volumeRender.nHeight = nHeight
volumeRender.nDepth = nDepth
volumeRender.windowTitle = "Quantum Turbulence  nPoints={0}".format(nPoints)
volumeRender.nTextures = 1
volumeRender.initGL()
#initialize pyCUDA context 
cudaDevice = setCudaDevice( devN=useDevice, usingAnimation=True )

#set thread grid for CUDA kernels
block_size_x, block_size_y, block_size_z = 8,8,8   #hardcoded, tune to your needs
gridx = nWidth // block_size_x + 1 * ( nWidth % block_size_x != 0 )
gridy = nHeight // block_size_y + 1 * ( nHeight % block_size_y != 0 )
gridz = nDepth // block_size_z + 1 * ( nDepth % block_size_z != 0 )
block3D = (block_size_x, block_size_y, block_size_z)
grid3D = (gridx, gridy, gridz)
nBlocks3D = grid3D[0]*grid3D[1]*grid3D[2]

print "\nCompiling CUDA code"
cudaCodeFile = open("cudaQturbulence.cu","r")
cudaCodeString_raw = cudaCodeFile.read().replace( "cudaP", cudaP ) 
cudaCodeString = cudaCodeString_raw % { "THREADS_PER_BLOCK":block3D[0]*block3D[1]*block3D[2], "B_WIDTH":block3D[0], "B_HEIGHT":block3D[1], "B_DEPTH":block3D[2] }
cudaCode = SourceModule(cudaCodeString)
getAlphas = cudaCode.get_function( "getAlphas_kernel" )
solvePartialX = cudaCode.get_function( "solvePartialX_kernel" )
solvePartialY = cudaCode.get_function( "solvePartialY_kernel" )
setBoundryConditionsKernel = cudaCode.get_function( 'setBoundryConditions_kernel' )
implicitStep1 = cudaCode.get_function( "implicitStep1_kernel" )
implicitStep2 = cudaCode.get_function( "implicitStep2_kernel" )
findActivityKernel = cudaCode.get_function( "findActivity_kernel" )
getActivityKernel = cudaCode.get_function( "getActivity_kernel" )
getVelocityKernel = cudaCode.get_function( "getVelocity_kernel" )
########################################################################
from pycuda.elementwise import ElementwiseKernel
########################################################################
multiplyByScalarReal = ElementwiseKernel(arguments="cudaP a, cudaP *realArray".replace("cudaP", cudaP),
				operation = "realArray[i] = a*realArray[i] ",
				name = "multiplyByScalarReal_kernel")
########################################################################
multiplyByScalarComplex = ElementwiseKernel(arguments="cudaP a, pycuda::complex<cudaP> *psi".replace("cudaP", cudaP),
				operation = "psi[i] = a*psi[i] ",
				name = "multiplyByScalarComplex_kernel",
				preamble="#include <pycuda-complex.hpp>")
getModulo = ElementwiseKernel(arguments="pycuda::complex<cudaP> *psi, cudaP *psiMod".replace("cudaP", cudaP),
			      operation = "cudaP mod = abs(psi[i]);\
					    psiMod[i] = mod*mod;".replace("cudaP", cudaP),	
			      name = "getModulo_kernel",
			      preamble="#include <pycuda-complex.hpp>")
########################################################################
sendModuloToUCHAR = ElementwiseKernel(arguments="cudaP *psiMod, unsigned char *psiUCHAR".replace("cudaP", cudaP),
			      operation = "psiUCHAR[i] = (unsigned char) ( -255*(psiMod[i]-1));",
			      name = "sendModuloToUCHAR_kernel")
########################################################################
getNorm = ReductionKernel( np.dtype(cudaPre),
			    neutral = "0",
			    arguments=" cudaP dx, cudaP dy, cudaP dz, pycuda::complex<cudaP> * psi ".replace("cudaP", cudaP),
			    map_expr = "( conj(psi[i])* psi[i] )._M_re*dx*dy*dz",
			    reduce_expr = "a+b",
			    name = "getNorm_kernel",
			    preamble="#include <pycuda-complex.hpp>")
########################################################################
def gaussian3D(x, y, z, gammaX=1, gammaY=1, gammaZ=1, random=False):    
  values =  np.exp( -gammaX*x*x - gammaY*y*y - gammaZ*z*z ).astype( cudaPre )
  if random:
    values += ( 100*np.random.random(values.shape) - 50 ) * values
  return values
########################################################################
def normalize( dx, dy, dz, complexArray ):
  factor = cudaPre( 1./(np.sqrt(getNorm(  dx, dy, dz, complexArray ).get())) )  #OPTIMIZATION
  multiplyByScalarComplex( factor, complexArray )
########################################################################
def implicit_iteration( ):
  global alpha
  #Make FFT
  fftPlan.execute( psi_d, psiFFT_d )
  #get Derivatives
  solvePartialX( Lx, psiFFT_d, partialX_d, fftKx_d, block=block3D, grid=grid3D) 
  solvePartialY( Ly, psiFFT_d, partialY_d, fftKy_d, block=block3D, grid=grid3D) 
  fftPlan.execute( partialX_d, inverse=True )
  fftPlan.execute( partialY_d, inverse=True )   
  implicitStep1( xMin, yMin, zMin, dx, dy, dz, alpha,  omega,  gammaX,  gammaY,  gammaZ,
		  partialX_d, partialY_d, psi_d, G_d, x0, y0, grid=grid3D, block=block3D)
  fftPlan.execute( G_d )
  implicitStep2( dt, fftKx_d , fftKy_d, fftKz_d, alpha, psiFFT_d, G_d, block=block3D, grid=grid3D) 
  fftPlan.execute( psiFFT_d, psi_d, inverse=True)  
  #setBoundryConditionsKernel( np.int32(nWidth), np.int32(nHeight), np.int32(nDepth), psi_d, block=block3D, grid=grid3D)  
  normalize(dx, dy, dz, psi_d)
  #GetAlphas
  getAlphas( dx, dy, dz, xMin, yMin, zMin, gammaX, gammaY, gammaZ, psi_d, alphas_d, block = block3D, grid=grid3D)
  alpha= cudaPre( ( 0.5*(gpuarray.max(alphas_d) + gpuarray.min(alphas_d)) ).get() )  #OPTIMIZACION 
########################################################################
def imaginaryStep():
  getModulo( psi_d, psiMod_d )
  factor = cudaPre(1./((gpuarray.max(psiMod_d)).get()))
  multiplyByScalarReal( factor, psiMod_d )
  sendModuloToUCHAR( psiMod_d, plotData_d)
  if plottingActive: 
    cuda.memset_d8(activity_d.ptr, 0, nBlocks3D )
    findActivityKernel( cudaPre(0.001), psi_d, activity_d, grid=grid3D, block=block3D )
    if plotVar == 0: getActivityKernel( psiOther_d, activity_d, grid=grid3D, block=block3D )
    if plotVar == 1:
      getVelocityKernel( np.int32(neighbors), dx, dy, dz, psi_d, activity_d, psiOther_d, grid=grid3D, block=block3D )
      factor = cudaPre(1./((gpuarray.max(psiOther_d)).get()))
      multiplyByScalarReal( factor, psiOther_d )
    sendModuloToUCHAR( psiOther_d, plotData_d)
  copyToScreenArray()
  [ implicit_iteration() for i in range(1) ]
########################################################################


print "\nInitializing Data"  
initialMemory = getFreeMemory( show=True )
psi_h = np.zeros( X.shape, dtype=cudaPreComplex )
psi_h.real = gaussian3D ( X, Y, Z, gammaX, gammaY, gammaZ, random=True ) 
print " Making FFT plan"
from pyfft.cuda import Plan
fftPlan = Plan((nDepth, nHeight, nWidth),  dtype=cudaPreComplex)  
fftKx_h = np.zeros( nWidth, dtype=cudaPre )
fftKy_h = np.zeros( nHeight, dtype=cudaPre )
fftKz_h = np.zeros( nDepth, dtype=cudaPre )
for i in range(nWidth/2):
  fftKx_h[i] = i*2*np.pi/Lx
for i in range(nWidth/2, nWidth):
  fftKx_h[i] = (i-nWidth)*2*np.pi/Lx  
for i in range(nHeight/2):
  fftKy_h[i] = i*2*np.pi/Ly
for i in range(nHeight/2, nHeight):
  fftKy_h[i] = (i-nHeight)*2*np.pi/Ly
for i in range(nDepth/2):
  fftKz_h[i] = i*2*np.pi/Lz
for i in range(nDepth/2, nDepth):
  fftKz_h[i] = (i-nDepth)*2*np.pi/Lz
psi_d = gpuarray.to_gpu(psi_h)
alphas_d = gpuarray.to_gpu(  np.zeros_like(psi_h.real) )
normalize( dx, dy, dz, psi_d )
getAlphas( dx, dy, dz, xMin, yMin, zMin, gammaX, gammaY, gammaZ, psi_d, alphas_d, block = block3D, grid=grid3D)
alpha=( 0.5*(gpuarray.max(alphas_d) + gpuarray.min(alphas_d)) ).get()  
psiMod_d = gpuarray.to_gpu(  np.zeros_like(psi_h.real) )
psiFFT_d = gpuarray.to_gpu(  np.zeros_like(psi_h) )
partialX_d = gpuarray.to_gpu(  np.zeros_like(psi_h) )
partialY_d = gpuarray.to_gpu(  np.zeros_like(psi_h) )
G_d = gpuarray.to_gpu(  np.zeros_like(psi_h) )
#Not really needed ( future improvement  ) 
fftKx_d = gpuarray.to_gpu( fftKx_h )         #OPTIMIZATION
fftKy_d = gpuarray.to_gpu( fftKy_h )
fftKz_d = gpuarray.to_gpu( fftKz_h )
activity_d = gpuarray.to_gpu( np.zeros( nBlocks3D, dtype=np.uint8 ) )
psiOther_d = gpuarray.to_gpu(  np.zeros_like(psi_h.real) )
#memory for plotting
#plotDataFloat_d = gpuarray.to_gpu(np.zeros_like(psi_h.real).astype(np.float32))
plotData_d = gpuarray.to_gpu(np.zeros([nDepth, nHeight, nWidth], dtype = np.uint8))
volumeRender.plotData_dArray, copyToScreenArray = gpuArray3DtocudaArray( plotData_d )
print "Total Global Memory Used: {0:.2f} MB\n".format(float(initialMemory-getFreeMemory( show=False ))/1e6) 

def keyboard(*args):
  #global volumeRender.transferScale, volumeRender.brightness, volumeRender.density, volumeRender.transferOffset
  global plottingActive
  ESCAPE = '\033'
  # If escape is pressed, kill everything.
  if args[0] == ESCAPE:
    print "Ending Simulation"
    #cuda.gl.Context.pop()
    sys.exit()
  if args[0] == '1':
    volumeRender.transferScale += np.float32(0.01)
    print "Image Transfer Scale: ", volumeRender.transferScale
  if args[0] == '2':
    volumeRender.transferScale -= np.float32(0.01)
    print "Image Transfer Scale: ",volumeRender.transferScale
  if args[0] == '4':
    volumeRender.brightness -= np.float32(0.1)
    print "Image Brightness : ",volumeRender.brightness
  if args[0] == '5':
    volumeRender.brightness += np.float32(0.1)
    print "Image Brightness : ",volumeRender.brightness
  if args[0] == '7':
    volumeRender.density -= np.float32(0.01)
    print "Image Density : ",volumeRender.density    
  if args[0] == '8':
    volumeRender.density += np.float32(0.01)
    print "Image Density : ",volumeRender.density    
  if args[0] == '3':
    volumeRender.transferOffset += np.float32(0.01)
    print "Image Offset : ", volumeRender.transferOffset    
  if args[0] == '6':
    volumeRender.transferOffset -= np.float32(0.01)
    print "Image Offset : ", volumeRender.transferOffset 
  if args[0] == 'a': 
    plottingActive = not plottingActive
    if plottingActive: print "plottingActive"

def specialKeyboardFunc( key, x, y ):
  global plotVar, neighbors
  if key== volumeRender.GLUT_KEY_UP:
    neighbors += 1
    if neighbors == 3: neighbors = 1
    print "Neighbors: ", neighbors
  #if key== volumeRender.GLUT_KEY_DOWN:
    #plotVar -= 1
    #if plotVar == -1: plotVar = 1
  if key== volumeRender.GLUT_KEY_RIGHT:
    plotVar += 1
    if plotVar == 2: plotVar = 0
  if key== volumeRender.GLUT_KEY_LEFT:
    plotVar -= 1
    if plotVar == -1: plotVar = 1    
  
  
#configure volumeRender functions 
volumeRender.viewTranslation[2] = -2
volumeRender.keyboard = keyboard
volumeRender.specialKeys = specialKeyboardFunc
volumeRender.stepFunc = imaginaryStep

#imaginaryStep()

#run volumeRender animation
volumeRender.animate()

