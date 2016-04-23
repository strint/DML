#include "lr.h"
#include "mpi.h"
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <iomanip>
#include "gtest/gtest.h"
#include "config.h"

int main(int argc,char* argv[]){  
    std::cout<<"cmd: ./train -trainfile -testfile"<<std::endl;
    int rank, nproc, namelen=100;
    MPI_Request req = {};
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    MPI_Status status;

    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
    MPI_Get_processor_name(processor_name, &namelen);
    //exec by main thread    
    const char *train_data_file = argv[1];
    char *test_data_file = argv[2];
    std::string split_tag = " ";
    
    load_data(train_data_file, split_tag);
    run(nproc, rank);
    MPI::Finalize();
    return 0;
}
