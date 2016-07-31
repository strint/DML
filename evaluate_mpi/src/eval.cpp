#include "eval.h"

Eval::Eval(Load_Data* ld) : data(ld){
    init();
}
Eval::~Eval(){
}

void Eval::init(){
    MAX_BUF_SIZE = 2048;
    g_all_non_clk = new float[data->MAX_ARRAY_SIZE];
    g_all_clk = new float[data->MAX_ARRAY_SIZE];
    g_nclk = new float[data->MAX_ARRAY_SIZE];
    g_clk = new float[data->MAX_ARRAY_SIZE];
}
int Eval::merge_clk(){
    bzero(g_nclk, data->MAX_ARRAY_SIZE * sizeof(float));
    bzero(g_clk, data->MAX_ARRAY_SIZE * sizeof(float));
    int cnt = data->clkinfo_list.size();
    for(int i = 0; i < cnt; i++){
        int idx = data->clkinfo_list[i].idx;
        g_nclk[idx] += data->clkinfo_list[i].nclk;
        g_clk[idx] += data->clkinfo_list[i].clk;
    }
    return 0;
}

int Eval::auc_cal(float* all_clk, float* all_nclk, double& auc_res){
    double clk_sum = 0.0;
    double nclk_sum = 0.0;
    double old_clk_sum = 0.0;
    double clksum_multi_nclksum = 0.0;
    double auc = 0.0;
    auc_res = 0.0;
    for(int i = 0; i < data->MAX_ARRAY_SIZE; i++){
        old_clk_sum = clk_sum;
        clk_sum += all_clk[i];
        nclk_sum += all_nclk[i];
        auc += (old_clk_sum + clk_sum) * all_nclk[i] / 2;
    }
    clksum_multi_nclksum = clk_sum * nclk_sum;
    auc_res = auc/(clksum_multi_nclksum);
}

int Eval::mpi_auc(int nprocs, int rank, double& auc){
    MPI_Status status;
    if(rank != MASTER_ID){
        MPI_Send(g_nclk, data->MAX_ARRAY_SIZE, MPI_FLOAT, MASTER_ID, MPI_NON_CLK_TAG, MPI_COMM_WORLD);
        MPI_Send(g_clk, data->MAX_ARRAY_SIZE, MPI_FLOAT, MASTER_ID, MPI_CLK_TAG, MPI_COMM_WORLD);
    }
    else if(rank == MASTER_ID){
        for(int i = 0; i < data->MAX_ARRAY_SIZE; i++){
            g_all_non_clk[i] = g_nclk[i];
            g_all_clk[i] = g_clk[i];
        }
        for(int i = 1; i < nprocs; i++){
            MPI_Recv(g_nclk, data->MAX_ARRAY_SIZE, MPI_FLOAT, i, MPI_NON_CLK_TAG, MPI_COMM_WORLD, &status);
            MPI_Recv(g_clk, data->MAX_ARRAY_SIZE, MPI_FLOAT, i, MPI_CLK_TAG, MPI_COMM_WORLD, &status);
            for(int i = 0; i < data->MAX_ARRAY_SIZE; i++){
                g_all_non_clk[i] += g_nclk[i];
                g_all_clk[i] += g_clk[i];
            }
        }
        auc_cal(g_all_non_clk, g_all_clk, auc);
    }
}

int Eval::mpi_peval(int nprocs, int rank, char* auc_file){
    double total_clk = 0.0;
    double total_nclk = 0.0;
    double auc = 0.0;
    double total_auc = 0.0;

    FILE *fp = NULL;
    merge_clk();
    mpi_auc(nprocs, rank, auc);

    if(MASTER_ID == rank){
        printf("AUC = %lf\n", auc);
    }
}
