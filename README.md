
# Accelerate a sparse convolutional layer with CUDA. 
## Using Coo format
```
==19186== Profiling result:
Time(%)      Time     Calls       Avg       Min       Max  Name
 96.74%  27.827ms         1  27.827ms  27.827ms  27.827ms  convLayerGPU_FAST(short*, int*, int*, int*)
  2.42%  696.02us         6  116.00us  87.230us  173.21us  [CUDA memcpy HtoD]
  0.56%  161.47us         1  161.47us  161.47us  161.47us  MaxPoolingGPU(int*, int*)
  0.28%  80.190us         1  80.190us  80.190us  80.190us  [CUDA memcpy DtoH]

==19186== API calls:
Time(%)      Time     Calls       Avg       Min       Max  Name
 71.38%  73.878ms         8  9.2348ms  3.8410us  73.455ms  cudaMalloc
 27.17%  28.119ms         1  28.119ms  28.119ms  28.119ms  cudaDeviceSynchronize
  1.10%  1.1386ms         7  162.65us  89.676us  189.69us  cudaMemcpy
  0.23%  240.68us        83  2.8990us     698ns  81.225us  cuDeviceGetAttribute
  0.03%  34.851us         2  17.425us  8.4510us  26.400us  cudaLaunch
  0.03%  31.569us         1  31.569us  31.569us  31.569us  cuDeviceTotalMem
  0.03%  27.238us         1  27.238us  27.238us  27.238us  cuDeviceGetName
  0.01%  12.362us         5  2.4720us  1.1870us  7.1240us  cudaFree
  0.01%  8.4500us         6  1.4080us     698ns  4.4690us  cudaSetupArgument
  0.00%  2.5840us         2  1.2920us     838ns  1.7460us  cuDeviceGetCount
  0.00%  2.3750us         2  1.1870us     838ns  1.5370us  cudaConfigureCall
  0.00%  2.0940us         2  1.0470us     977ns  1.1170us  cuDeviceGet

```

- **Total Result**
```
  ================ Result ===================
CPU time for executing a typical convolutional layer = 16631.2ms
GPU time for executing a typical convolutional layer = 113.107ms
Congratulations! You pass the check.
Speedup: 147.039
=====================================================
 
 ================ Result ===================
CPU time for executing a typical convolutional layer = 16609.3ms
GPU time for executing a typical convolutional layer = 128.902ms
Congratulations! You pass the check.
Speedup: 128.853
```
** result : 1.14 times faster


## Useful Reference

### Part-II
* Network pruning: [Learning both Weights and Connections for Efficient Neural Networks](https://arxiv.org/pdf/1506.02626.pdf)
* Sparsity in Neurons: [Cnvlutin: Ineffectual-neuron-free Deep Neural Network Computing](http://www.ece.ubc.ca/~aamodt/papers/Cnvlutin.ISCA2016.pdf)
* Sparse data GPU: [Implementing Sparse Matrix-Vector Multiplication on Throughput-Oriented Processors](https://pdfs.semanticscholar.org/9abb/086fabdcd2853ed8303c0f9a62cf4b917a62.pdf)
* Sparse data with CUDA: [Efficient Sparse Matrix-Vector Multiplication on CUDA](http://wnbell.com/media/2008-12-NVR-SpMV/nvr-2008-004.pdf)

