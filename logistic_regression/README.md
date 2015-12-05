# 并行逻辑回归

## 依赖环境配置
### CBLAS编译安装与使用举例
* http://www.linuxidc.com/Linux/2015-02/113169.htm
* http://www.netlib.org/blas/

### OpenBLAS编译和安装简介
* http://www.leexiang.com/how-to-compile-and-use-openblas
* make install with default path

### 错误解决办法
```
think@think-ubt:~/x/g/DML/logistic_regression$ mpiexec -mca btl ^openib -np 2 ./train
./train: error while loading shared libraries: libopenblas.so.0: cannot open shared object file: No such file or directory
```
* error while loading shared libraries的解決方法 http://blog.csdn.net/dumeifang/article/details/2963223
* the shared library is in /opt/OpenBLAS/lib

