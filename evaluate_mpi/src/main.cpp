#include "eval.h"
#include <stdlib.h>
#define MAX_FILENAME_LEN 4096

int main(int argc, char* argv[]){
    int rank = 0, nproc = 0;
    char score_file[MAX_FILENAME_LEN];
    char* auc_file = NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    snprintf(score_file, sizeof(score_file), "%s", argv[1]);
    
    if(argc == 3 && rank == MASTER_ID){
        auc_file = (char*)malloc(MAX_FILENAME_LEN * sizeof(char));
        snprintf(auc_file, MAX_FILENAME_LEN, "%s", argv[2]);
    }
    else auc_file = NULL;
    Load_Data ld;
    ld.load_pctr_nclk_clk(score_file, rank);
    Eval eval(&ld);
    eval.mpi_peval(nproc, rank, auc_file);
    
    MPI_Finalize();
    return 0;
}
