#include "cuda_runtime.h" // CUDA utilities and system includes
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include <cutil.h>
#include <ctime>
#include <helper_timer.h>
#include <timer.h>
#include <thread>

using namespace std; 

#include <helper_functions.h> // includes for SDK helper functions
#include <helper_cuda.h>      // includes for cuda initialization and error checking

StopWatchInterface *timer = NULL;

#define BLOCK_SIZE 16
#define FILTER_SIZE 3
#define TILE_SIZE 16 

void Anisodiff(unsigned char *im_in, unsigned char *im_out, int col , int row, double lambda, int opt, int kappa) {
	//This algorithm approximates the solution of the heat (Perona-Malik anisotropic) diffusion equation
//Adapted and modified from Peter Kovesi's MATLAB code implementation of AD.
	int hN[3][3] = {0, 1, 0, 0, -1, 0, 0, 0, 0};
	int hS[3][3] = {0, 0, 0, 0, -1, 0, 0, 1, 0};
	int hE[3][3] = {0, 0, 0, 0, -1, 1, 0, 0, 0};
	int hW[3][3] = {0, 0, 0, 1, -1, 0, 0, 0, 0};
	int offset = 1 * width ; //int val = 0;
#pragma omp parallel for
	for (int row =1; row < height -1; row ++){
#pragma omp parallel for
		for (int col =1; col <width -1; col ++){
			int deltaN , deltaS, deltaE, deltaW;
			deltaN = deltaS = deltaE = deltaW = 0;
			double cN, cS, cE, cW; //cN = cS = cE = cW = 0;
#pragma omp parallel for
			for (int i= -1; i <= 1; i++){
#pragma omp parallel for
				for (int j= -1; j <=1; j++){
					int xn = im_in [( row + j) * width + (col + i)];
					deltaN += xn * hN[i+1][j+1];
					deltaS += xn * hS[i+1][j+1];
					deltaE += xn * hE[i+1][j+1];
					deltaW += xn * hW[i+1][j+1];}}
			////// Conduction
			if (opt == 1) {
				cN = exp(-powf((deltaN / kappa), 2));
				cS = exp(-powf((deltaS / kappa), 2));
				cE = exp(-powf((deltaE / kappa), 2));
				cW = exp(-powf((deltaW / kappa), 2));
			} else if (opt == 2) {
				cN = 1 / (1 + powf((deltaN / kappa), 2));
				cS = 1 / (1 + powf((deltaS / kappa), 2));
				cE = 1 / (1 + powf((deltaE / kappa), 2));
				cW = 1 / (1 + powf((deltaW / kappa), 2));
			}
			int u0 = im_in[row*width+col];
			u0 = u0+lambda*(cN * deltaN + cS * deltaS + cE * deltaE + cW * deltaW);
			if (u0 > 255) u0 = 255;
			if (u0 < 0) u0 = 0; 	
			im_out [offset + col] = u0; }
		offset += width;}			
}


__global__ void CUDA_Anisodiff(unsigned char *im_in, unsigned char *im_out, int height , int width, double lambda, int opt, int kappa){
//This algorithm approximates the solution of the heat (Perona-Malik anisotropic) diffusion equation
//Adapted and modified from Peter Kovesi's MATLAB code implementation of AD.
	int col = blockIdx .x * blockDim .x + threadIdx .x;
	int row = blockIdx .y * blockDim .y + threadIdx .y;

	int hN[3][3] = {0, 1, 0, 0, -1, 0, 0, 0, 0};
	int hS[3][3] = {0, 0, 0, 0, -1, 0, 0, 1, 0};
	int hE[3][3] = {0, 0, 0, 0, -1, 1, 0, 0, 0};
	int hW[3][3] = {0, 0, 0, 1, -1, 0, 0, 0, 0};
	int deltaN , deltaS, deltaE, deltaW;
	deltaN = deltaS = deltaE = deltaW = 0;
	double cN, cS, cE, cW; //cN = cS = cE = cW = 0;
	int u0 = 0;

	for (int i= -1; i <= 1; i++){
		for (int j= -1; j <=1; j++){
			int xn = im_in [( row + j) * width + (col + i)];
			deltaN += xn * hN[i+1][j+1];
			deltaS += xn * hS[i+1][j+1];
			deltaE += xn * hE[i+1][j+1];
			deltaW += xn * hW[i+1][j+1];}}
	////// Conduction
	if (opt == 1) {
		cN = exp(-powf((deltaN / kappa), 2));  //use powf(x,y) or sqrtf(x) to avoid "error: calling a host function from a device/global function is not allowed"
		cS = exp(-powf((deltaS / kappa), 2)); //works well too in CPU
		cE = exp(-powf((deltaE / kappa), 2));
		cW = exp(-powf((deltaW / kappa), 2));
	} else if (opt == 2) {
		cN = 1 / (1 + powf((deltaN / kappa), 2));
		cS = 1 / (1 + powf((deltaS / kappa), 2));
		cE = 1 / (1 + powf((deltaE / kappa), 2));
		cW = 1 / (1 + powf((deltaW / kappa), 2));
	}
	u0 = im_in[row*width+col];

	u0 = u0+lambda*(cN * deltaN + cS * deltaS + cE * deltaE + cW * deltaW);
	
	if (u0 > 255) u0 = 255;
	if (u0 < 0) u0 = 0; 
	im_out [row * width + col ] = u0; //
}


int main (int argc , char **argv ) {
	/* ******************** Prepare job ****************************/
	unsigned char *d_im_outPixels ;
	unsigned char *h_im_outPixels ;
	unsigned char *h_pixels = NULL ;
	unsigned char *d_pixels = NULL ; 

	char *srcPath = "<enter your own image pgm file path>";
	char *h_im_outPath = "<enter your own PC processed image pgm file path>";
	char *d_im_outPath = "<enter your own GPU processed image pgm file path>";*/

	sdkLoadPGM<unsigned char>(srcPath, & h_pixels , &width , & height);
	int ImageSize = sizeof(unsigned char) * width * height;
	h_im_outPixels = (unsigned char *) malloc(ImageSize);
	cudaMalloc(( void **)& d_pixels , ImageSize );
	cudaMalloc(( void **)& d_im_outPixels , ImageSize );
	cudaMemcpy(d_pixels , h_pixels , ImageSize , cudaMemcpyHostToDevice);
	/* ******************** END Prepare job*************************** */

	int iter = 10; 
	double lambda = 0.25;
	int kappa = 10; 
	int opt = 2;	

	/* ************************ Host processing************************* */
	float time;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	clock_t starttime , endtime , difference ;
	printf (" Starting host processing \n");
	starttime = clock ();
	
	norm_array(h_pixels, height, width);

	for (int n = 1; n < iter; ++n) {
		CUDA_Anisodiff(h_pixels, h_im_outPixels, height, width, lambda, opt, kappa); 
		h_pixels = h_im_outPixels;
	}	
	scale_array(h_im_outPixels, height, width);
	uint8_array(h_im_outPixels, height, width);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf("CPU - elapsed time:  %3.3f ms \n", time);

	sdkSavePGM<unsigned char>(h_im_outPath, h_im_outPixels, width , height);
	///* ************************ END Host processing************************* */


	///* ************************ Device processing************************* */
	dim3 block(BLOCK_SIZE,BLOCK_SIZE) ;
	dim3 grid(width/BLOCK_SIZE, height/BLOCK_SIZE) ;
	sdkCreateTimer(&timer); 
	sdkStartTimer(&timer);
	//* CUDA method */
	CUDA_norm_array<<<grid, block>>>(d_pixels, height, width); //only use for TV-L1/L2 algorithm


	for (int n = 1; n < iter; ++n) {
		CUDA_Anisodiff<<<grid, block>>>(d_pixels, d_im_outPixels, height, width, lambda, opt, kappa); 
		d_pixels = d_im_outPixels;
	}

	cudaThreadSynchronize();
	sdkStopTimer(&timer);
	printf("CUDA execution time = %3.3fms\n", sdkGetTimerValue (&timer));
	cudaMemcpy(h_im_outPixels, d_im_outPixels, ImageSize, cudaMemcpyDeviceToHost);
	cudaFree(d_pixels);
	cudaFree(d_im_outPixels);
	sdkSavePGM<unsigned char>(d_im_outPath, h_im_outPixels , width , height);
	sdkResetTimer(&timer);

	/* ************************ END Device processing************************* */
	printf("Press enter to exit ...\n");
	getchar();
}
