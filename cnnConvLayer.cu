// This program executes a typical convolutional layer in regular CNNs
<<<<<<< HEAD
//Faster format
=======
//in CSR format


>>>>>>> 001d4846d042f14babfdf9d9b8fba6a811d138e5
#include <iostream>
#include "cnnConvLayer.h"
#include <stdio.h>
#include <unistd.h>
using namespace std;

#define xDim 512
#define yDim 32
#define zDim 32

#define xThreadDim 4
#define yThreadDim 16
#define zThreadDim 16

#define Pool_xDim 512
#define Pool_yDim 16
#define Pool_zDim 16


int outputsize = 512*16*16;
int Outputsize = xDim*yDim*zDim;

int *devoutNeu;
int *devPooling;
short *devFilt;
short *devinNeu;


/*COO Format*/
short *devfiltCooNNZ;
short *devfiltCooData;
short *devfiltCooRow;
short *devfiltCooCol;

short *devinNeuCooNNZ;
short *devinNeuCooData;
short *devinNeuCooRow;
short *devinNeuCooCol;


int *devfiltFastData;

/*COO Format*/
short *devfiltCooNNZ;
short *devfiltCooData;
short *devfiltCooRow;
short *devfiltCooCol;

short *devinNeuCooNNZ;
short *devinNeuCooData;
short *devinNeuCooRow;
short *devinNeuCooCol;


int *outResult = new int[outputsize]();
int *outResult_neu = new int[Outputsize]();
int *filtFastData = new int [FILTNUM*FMDEPTH]();


// This is the CPU version, please don't modify it
void convLayerCPU()
{
	// declarations for bunch of indexing parameters
	int fn, sli, fmy, fmx, y, x;
	int sum, ifmy, ifmx, ofmy, ofmx;
	int filtIdx, inNeuIdx, outNeuIdx, outIdx;
	int filtVol = FMDEPTH * FILTSIZE * FILTSIZE;
	int filtArea = FILTSIZE * FILTSIZE;
	int fmArea = FMSIZE *FMSIZE;
	int outArea = FMSIZE/2 * FMSIZE/2;


	cout << "convolutioning..." << endl;

	// Convolution
	for(fn = 0; fn < FILTNUM; fn++) //512
	{
		for(fmy = 0; fmy < FMSIZE; fmy += STRIDE) //32
		{
			for(fmx = 0; fmx < FMSIZE; fmx += STRIDE)  //32
			{
				


				sum = 0;
				for(sli = 0; sli < FMDEPTH; sli++)  //512
				{
					for(y = 0; y < FILTSIZE; y++)  //3
					{
						for(x = 0; x < FILTSIZE; x++)  //3
						{
							ifmy = fmy - FILTSIZE / 2 + y;		//no dependancy
							ifmx = fmx - FILTSIZE / 2 + x;		//no dependancy
							filtIdx = (fn * filtVol) + (sli * filtArea) + (y * FILTSIZE) + x;	//no dependancy
							inNeuIdx = sli*fmArea + ifmy*FMSIZE + ifmx;							//no dependancy
							if(ifmy >= 0 && ifmy < FMSIZE && ifmx >= 0 && ifmx < FMSIZE)
								sum += filt[filtIdx] * inNeu[inNeuIdx];
							//if(fn == 0 && fmx == 0 && fmy == 0 && sli <10)
							//	printf("filt[%d] = %d\n",filtIdx,filt[filtIdx] );
						}
					}
				}



				// Activation - ReLU
				outNeuIdx = fn*fmArea + fmy*FMSIZE + fmx;
				if(sum <= 0)
					outNeu[outNeuIdx] = 0;
				else
					outNeu[outNeuIdx] = sum;
			}
		}
	}


 	cout << "Pooling....." << endl;
	// Max Pooling with Window Size 2x2
	int max, tmpVal;
	for(sli = 0; sli < FILTNUM; sli++)
	{
		for(fmy = 0; fmy < FMSIZE/2 ; fmy += 1)
		{
			for(fmx = 0; fmx < FMSIZE/2 ; fmx += 1)
			{
				outNeuIdx = sli*fmArea + fmy*2*FMSIZE + fmx*2;
				max = outNeu[outNeuIdx];
				for(y = 0; y < 2; y++)
				{
					for(x = 0; x < 2; x++)
					{
						ofmy = fmy*2 + y;
						ofmx = fmx*2 + x;
						outNeuIdx = sli*fmArea + ofmy*FMSIZE + ofmx;
						tmpVal = outNeu[outNeuIdx];
						if(tmpVal > max)
							max = tmpVal;
					}
				}
				outIdx = sli*outArea + fmy*FMSIZE/2 + fmx;
				outCPU[outIdx] = max;
			}
		}
	}
}


/* 	Fast Format:

	filtFastData[i] = Data[i] + Row[i] + Col[i]
*/
void initFastFormat()
{
	int tempdata = 0;

	for (int j = 0; j < FILTNUM*FMDEPTH; j++)
		filtFastData[j] = 0;

	for (int i = 0; i < FILTNUM*FMDEPTH; i++)
	{
		tempdata = filtCooData[i];
		filtFastData[i] = tempdata*100; 
		filtFastData[i] += filtCooRow[i]*10; 
		filtFastData[i] += filtCooCol[i];
	}
}

void checkFormat()
{
	int data, row, col;
	int *temp = new int [FILTNUM*FMDEPTH]();

	for (int i = 0; i < FILTNUM*FMDEPTH; ++i)
	{
		temp[i] = filtFastData[i];

		col = temp[i]%10;
		temp[i] = (temp[i] - col)/10;
		row = temp[i]%10;
		temp[i] = (temp[i] - row)/10;
		data = temp[i];

		if(data != filtCooData[i])
		{
			printf("data wrong: %d to %d at index = %d\n",data,filtCooData[i],i );
			break;
		}
		else if(row != filtCooRow[i])
		{
			printf("row wrong: %d to %d at index = %d\n",row,filtCooRow[i],i );
			break;
		}
		else if(col != filtCooCol[i])
		{
			printf("col wrong: %d to %d at index = %d\n",col,filtCooCol[i],i );
			break;
		}
	}

	printf("Format checking Done!!\n");
}

void initGPU()
{
<<<<<<< HEAD
	int outNeuVol = FILTNUM * FMSIZE * FMSIZE;  		//512x32x32
	int outPolVol = FILTNUM * FMSIZE/2 * FMSIZE/2;  	//512x16x16
	int inNeuVol = sizeof(short)*FMDEPTH*FMSIZE*FMSIZE;	//512x32x32 

	//output from kernel 
	cudaMalloc(&devoutNeu, sizeof(int)*outNeuVol);	//int
	cudaMalloc(&devPooling, sizeof(int)*outPolVol);	//int
	cudaMalloc(&devinNeu, inNeuVol);	//input to kernel	//short  input to kernel

=======
	int outNeuVol = FILTNUM * FMSIZE * FMSIZE;  //512x32x32
	int outPolVol = FILTNUM * FMSIZE/2 * FMSIZE/2;  //512x16x16
	//int filtTensorVol = sizeof(short)*FILTNUM*FMDEPTH*FILTSIZE*FILTSIZE; //512x512x3x3
	int inNeuVol = sizeof(short)*FMDEPTH*FMSIZE*FMSIZE;	//512x32x32 
	int filtCOOVol = sizeof(short)*FILTNUM*FMDEPTH; //512x512x1

	//output from kernel 
	cudaMalloc(&devoutNeu, sizeof(int)*outNeuVol);
	cudaMalloc(&devPooling, sizeof(int)*outPolVol);
	
	//input to kernel
	cudaMalloc(&devinNeu, inNeuVol);
>>>>>>> 001d4846d042f14babfdf9d9b8fba6a811d138e5
	cudaMemcpy(devinNeu, inNeu, inNeuVol, cudaMemcpyHostToDevice);
}

<<<<<<< HEAD
void initCooMemoryCopy()
{
	int filtCOOVol = sizeof(short)*FILTNUM*FMDEPTH; 	//512x512x1

	cudaMalloc(&devfiltCooNNZ, filtCOOVol);	//short input COO to kernel
=======

	//input COO to kernel
	//cudaMalloc(&devfiltCooNNZ, filtCOOVol);
>>>>>>> 001d4846d042f14babfdf9d9b8fba6a811d138e5
	cudaMalloc(&devfiltCooData, filtCOOVol);
	cudaMalloc(&devfiltCooRow, filtCOOVol);
	cudaMalloc(&devfiltCooCol, filtCOOVol);

<<<<<<< HEAD
	cudaMemcpy(devfiltCooNNZ, filtCooNNZ, filtCOOVol, cudaMemcpyHostToDevice );
	cudaMemcpy(devfiltCooData, filtCooData, filtCOOVol, cudaMemcpyHostToDevice );
	cudaMemcpy(devfiltCooRow, filtCooRow, filtCOOVol, cudaMemcpyHostToDevice );
	cudaMemcpy(devfiltCooCol, filtCooCol, filtCOOVol, cudaMemcpyHostToDevice );
}

void initFASTMemoryCopy()
{
	cudaMalloc(&devfiltFastData, sizeof(int)*FILTNUM*FMDEPTH);
	cudaMemcpy(devfiltFastData, filtFastData, sizeof(int)*FILTNUM*FMDEPTH, cudaMemcpyHostToDevice);
=======
	//cudaMemcpy(devfiltCooNNZ, filtCooNNZ, filtCOOVol, cudaMemcpyHostToDevice );
	cudaMemcpy(devfiltCooData, filtCooData, filtCOOVol, cudaMemcpyHostToDevice );
	cudaMemcpy(devfiltCooRow, filtCooRow, filtCOOVol, cudaMemcpyHostToDevice );
	cudaMemcpy(devfiltCooCol, filtCooCol, filtCOOVol, cudaMemcpyHostToDevice );

	//cudaMemcpy(devoutNeu, outNeu,sizeof(int)*outNeuVol, cudaMemcpyHostToDevice ); // debug for race outNeu
>>>>>>> 001d4846d042f14babfdf9d9b8fba6a811d138e5
}


/***	Implement your CUDA Kernel here	***/
__global__
void convLayerGPU(short *InNeu, short *FiltCooData, short *FiltCooRow, short *FiltCooCol, int *outNeural, int *outPooling)
{
	int threadX = threadIdx.x + blockIdx.x * blockDim.x;
	int threadY = threadIdx.y + blockIdx.y * blockDim.y;
	int threadZ = threadIdx.z + blockIdx.z * blockDim.z;
<<<<<<< HEAD
	//int xall = blockDim.x * gridDim.x;
	//int yall = blockDim.y * gridDim.y;
	//int GlobalThreadId = threadX + threadY * xall + threadZ * xall * yall;
	//int GlobalBlockId = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.y * gridDim.x;

=======
>>>>>>> 001d4846d042f14babfdf9d9b8fba6a811d138e5
	int ifmy, ifmx;
	int inNeuIdx, outNeuIdx, CooIdx;
	int fmArea = 1024;	//32x32
	int sum = 0;

	for (int i = 0; i < 512; ++i)
	{
		CooIdx = threadX*512 + i;

		ifmy = threadY - 3 / 2 + FiltCooRow[CooIdx];		
		ifmx = threadZ - 3 / 2 + FiltCooCol[CooIdx];		
		inNeuIdx = i * fmArea + ifmy * 32 + ifmx;	
		if(ifmy >= 0 && ifmy < 32 && ifmx >= 0 && ifmx < 32)	
			sum += FiltCooData[CooIdx] * InNeu[inNeuIdx];
	}


	outNeuIdx = threadX * fmArea + threadY*32 + threadZ;
	if(sum <= 0)
		outNeural[outNeuIdx] = 0;
	else
		outNeural[outNeuIdx] = sum;
}


/***	Implement your CUDA Kernel here	***/
__global__
void convLayerGPU_FAST(short *InNeu, int *FiltFastData, int *outNeural, int *outPooling)
{
	int threadX = threadIdx.x + blockIdx.x * blockDim.x;
	int threadY = threadIdx.y + blockIdx.y * blockDim.y;
	int threadZ = threadIdx.z + blockIdx.z * blockDim.z;
	
	int ifmy, ifmx;
	int inNeuIdx, outNeuIdx, FastIdx;
	int fmArea = 1024;	//32x32
	int sum = 0;
	int row = 0;
	int col = 0;
	int data = 0;


	for (int i = 0; i < 512; ++i)
	{
		FastIdx = threadX*512 + i;
		data = FiltFastData[FastIdx];

		col = data % 10;
		data = (data - col)/10;
		row = data%10;
		data = (data - row)/10;


		ifmy = threadY - 3 / 2 + row;		
		ifmx = threadZ - 3 / 2 + col;		
		inNeuIdx = i * fmArea + ifmy * 32 + ifmx;	
		if(ifmy >= 0 && ifmy < 32 && ifmx >= 0 && ifmx < 32)	
			sum += data * InNeu[inNeuIdx];
	}

	// Activation - ReLU
	outNeuIdx = threadX * fmArea + threadY*32 + threadZ;
	if(sum <= 0)
		outNeural[outNeuIdx] = 0;
	else
		outNeural[outNeuIdx] = sum;
}



__global__
void MaxPoolingGPU(int *outNeural, int *outPooling)  // Max Pooling with Window Size 2x2
{
	int threadX = threadIdx.x + blockIdx.x * blockDim.x;
	int threadY = threadIdx.y + blockIdx.y * blockDim.y;
	int threadZ = threadIdx.z + blockIdx.z * blockDim.z;
<<<<<<< HEAD
=======

>>>>>>> 001d4846d042f14babfdf9d9b8fba6a811d138e5
	int max, tmpVal, outNeuIdx, x, y;
	int fmArea = 1024;
	int outArea = 256;
	int  ofmy, ofmx, outIdx; // pooling varable

	outNeuIdx = threadX*fmArea + threadY*2*32 + threadZ*2;
	max = outNeural[outNeuIdx];
	for(y = 0; y < 2; y++)
	{
		for(x = 0; x < 2; x++)
		{
			ofmy = threadY*2 + y;
			ofmx = threadZ*2 + x;
			outNeuIdx = threadX*fmArea + ofmy*32 + ofmx;
			tmpVal = outNeural[outNeuIdx];
			if(tmpVal > max)
				max = tmpVal;
		}
	}
	outIdx = threadX*outArea + threadY*32/2 + threadZ;
	outPooling[outIdx] = max;
}


int main()
{
	float convLayerCPUExecTime, convLayerGPUExecTime;
	init();
	initCoo();
<<<<<<< HEAD
	initFastFormat();
	checkFormat();


=======
	int outSize = sizeof(int)*outputsize;
>>>>>>> 001d4846d042f14babfdf9d9b8fba6a811d138e5

	timespec time_begin, time_end;
  	clock_gettime(CLOCK_REALTIME, &time_begin);
	convLayerCPU();
  	clock_gettime(CLOCK_REALTIME, &time_end);
	convLayerCPUExecTime = timespec_diff_us(time_begin, time_end);
	cout << " ================ Result ===================" << endl;
	cout << "CPU time for executing a typical convolutional layer = " <<  convLayerCPUExecTime / 1000 << "ms" << endl;


<<<<<<< HEAD
=======

>>>>>>> 001d4846d042f14babfdf9d9b8fba6a811d138e5
 	dim3 threadPerBlock(xThreadDim, yThreadDim, zThreadDim);
 	dim3 numBlocks(xDim/xThreadDim, yDim/yThreadDim, zDim/zThreadDim);
 	dim3 Pool_threadPerBlock(xThreadDim, yThreadDim, zThreadDim);
 	dim3 Pool_numBlocks(Pool_xDim/xThreadDim, Pool_yDim/yThreadDim, Pool_zDim/zThreadDim);

 	clock_gettime(CLOCK_REALTIME, &time_begin);
 	initGPU();
<<<<<<< HEAD
 	initCooMemoryCopy();
 	initFASTMemoryCopy();
	//convLayerGPU<<<numBlocks,threadPerBlock>>>(devinNeu, devfiltCooData, devfiltCooRow, devfiltCooCol, devoutNeu, devPooling);
	convLayerGPU_FAST<<<numBlocks,threadPerBlock>>>(devinNeu, devfiltFastData, devoutNeu, devPooling);
	MaxPoolingGPU<<<Pool_numBlocks , Pool_threadPerBlock>>>(devoutNeu, devPooling);
	cudaDeviceSynchronize();

	int outSize = sizeof(int)*outputsize;
	cudaMemcpy(outGPU, devPooling, outSize, cudaMemcpyDeviceToHost);
=======


	convLayerGPU<<<numBlocks,threadPerBlock>>>(devinNeu , devfiltCooData, devfiltCooRow, devfiltCooCol, devoutNeu, devPooling);
	MaxPoolingGPU<<<Pool_numBlocks , Pool_threadPerBlock>>>(devoutNeu, devPooling);
	cudaDeviceSynchronize();

	cudaMemcpy(outGPU, devPooling, outSize, cudaMemcpyDeviceToHost);


	clock_gettime(CLOCK_REALTIME, &time_end);
	convLayerGPUExecTime = timespec_diff_us(time_begin, time_end);
	cout << "GPU time for executing a typical convolutional layer = " << convLayerGPUExecTime / 1000 << "ms" << endl;


>>>>>>> 001d4846d042f14babfdf9d9b8fba6a811d138e5
	//int OutSize = sizeof(int)*Outputsize;
	//cudaMemcpy(outGlobalBarrier, devGlobalBarrier,OutSize, cudaMemcpyDeviceToHost );

	clock_gettime(CLOCK_REALTIME, &time_end);
	convLayerGPUExecTime = timespec_diff_us(time_begin, time_end);
	cout << "GPU time for executing a typical convolutional layer = " << convLayerGPUExecTime / 1000 << "ms" << endl;




	if(checker())
	{
		cout << "Congratulations! You pass the check." << endl;
		cout << "Speedup: " << (float)convLayerCPUExecTime / convLayerGPUExecTime << endl;
	}
	else
		cout << "Sorry! Your result is wrong." << endl;

	cout << "=====================================================" << endl;

	cudaFree(&devoutNeu);
	cudaFree(&devPooling);
	cudaFree(&devinNeu);
	cudaFree(&devfiltFastData);

	cudaFree(&devfiltCooNNZ);
	cudaFree(&devfiltCooData);
	cudaFree(&devfiltCooRow);
	cudaFree(&devfiltCooCol);

	cudaFree(&devinNeuCooNNZ);
	cudaFree(&devinNeuCooData);
	cudaFree(&devinNeuCooRow);
	cudaFree(&devinNeuCooCol);


	delete [] outResult;
	delete [] outResult_neu;
	delete [] filtFastData;
	ending();

	return 0;
}
