#include <mpi.h>
#include "load_data.h"

class Eval{
public:
    Eval(Load_Data* data);
    ~Eval();
    int mpi_peval(int nproc, int rank, char* auc_file);
    Load_Data* data;
private:
    void init();
    int merge_clk();
    int mpi_auc(int nprocs, int rank, double& auc);
    int auc_cal(float* all_clk, float* all_nclk, double& auc_res);

    int MAX_BUF_SIZE;
    float* g_all_non_clk;
    float* g_all_clk;
    float* g_nclk;
    float* g_clk;
    double g_total_clk;
    double g_total_nclk;
};
