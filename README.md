
# Accelerate a sparse convolutional layer with CUDA. 
## Using Coo format
```
Time(%)      Time     Calls       Avg       Min       Max  Name
 98.28%  43.557ms         1  43.557ms  43.557ms  43.557ms  convLayerGPU()
  1.20%  530.35us         5  106.07us  87.934us  176.54us  [CUDA memcpy HtoD]
  0.35%  153.76us         1  153.76us  153.76us  153.76us  MaxPoolingGPU(int*, int*)
  0.18%  80.190us         1  80.190us  80.190us  80.190us  [CUDA memcpy DtoH]

==3341== API calls:
Time(%)      Time     Calls       Avg       Min       Max  Name
 62.64%  76.045ms         7  10.864ms  2.7940us  75.734ms  cudaMalloc
 36.04%  43.750ms         1  43.750ms  43.750ms  43.750ms  cudaDeviceSynchronize
  1.01%  1.2206ms         6  203.44us  175.16us  263.65us  cudaMemcpy
  0.20%  243.33us        83  2.9310us     698ns  82.064us  cuDeviceGetAttribute
  0.04%  43.511us         2  21.755us  9.3590us  34.152us  cudaLaunch
  0.03%  31.568us         1  31.568us  31.568us  31.568us  cuDeviceTotalMem
  0.02%  27.587us         1  27.587us  27.587us  27.587us  cuDeviceGetName
  0.02%  19.345us         4  4.8360us  1.1870us  15.435us  cudaFree
  0.01%  12.571us        10  1.2570us     698ns  4.7490us  cudaSetupArgument
  0.00%  2.8640us         2  1.4320us     908ns  1.9560us  cudaConfigureCall
  0.00%  2.7230us         2  1.3610us     838ns  1.8850us  cuDeviceGetCount
  0.00%  1.7460us         2     873ns     768ns     978ns  cuDeviceGet
```

- **Total Result**
```
 ================ Result ===================
CPU time for executing a typical convolutional layer = 16609.3ms
GPU time for executing a typical convolutional layer = 128.902ms
Congratulations! You pass the check.
Speedup: 128.853
```


## Useful Reference

### Part-II
* Network pruning: [Learning both Weights and Connections for Efficient Neural Networks](https://arxiv.org/pdf/1506.02626.pdf)
* Sparsity in Neurons: [Cnvlutin: Ineffectual-neuron-free Deep Neural Network Computing](http://www.ece.ubc.ca/~aamodt/papers/Cnvlutin.ISCA2016.pdf)
* Sparse data GPU: [Implementing Sparse Matrix-Vector Multiplication on Throughput-Oriented Processors](https://pdfs.semanticscholar.org/9abb/086fabdcd2853ed8303c0f9a62cf4b917a62.pdf)
* Sparse data with CUDA: [Efficient Sparse Matrix-Vector Multiplication on CUDA](http://wnbell.com/media/2008-12-NVR-SpMV/nvr-2008-004.pdf)

