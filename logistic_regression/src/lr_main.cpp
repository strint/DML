#include "lr.h"
#include "mpi.h"
#include <string.h>
#include "gtest/gtest.h"
#include "config.h"

int main(int argc,char* argv[]){  
    int rank, nproc;

    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&nproc);
    const char *train_data_file = argv[1];
    const char *test_data_file = argv[2];
    std::string split_tag = " ";
    Load_Data ld; 
    ld.load_data(train_data_file, split_tag, rank, nproc);
    //LR lr;
    //lr.run(nproc, rank);
    MPI::Finalize();
    return 0;
}
