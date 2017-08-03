# 3D Elastic Benchmark
Test of fourth order elastic compute kernels. 

Implementations tested:
* [SeisCL](https://github.com/gfabieno/SeisCL)
* [AWP-ODC](https://github.com/HPGeoC/awp-odc-os)
* [WaveFlow](../problems/elastic3d.py)

**NOTE**: AWP-ODC includes attenuation, free-surface, and PML boundaries in the computation.


| Block size      | AWP-ODC | WaveFlow (Runtime Env) | SeisCL  | SeisCL (Attenuation) |
|-----------------|---------|------------------------|---------|----------------------|
| 64 x  64 x  64  | 1109.78 | 333.11                 | 2304.15 | 1923.08              |
| 128 x 128 x 128 | 236.20  | 55.60                  | 330.25  | 266.67               |
| 256 x 256 x 256 | 51.25   | 6.86                   | 37.5    | 30.15                |


Average number of iterations per second (fps) reported out of 1000 iterations.

**GPU**: NVIDIA Titan X (Pascal)
