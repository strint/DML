#include "owlqn.h"
#include "mpi.h"
#include <string.h>
//#include "gtest/gtest.h"
#include "config.h"

int main(int argc,char* argv[]){  
    int rank, nproc;

    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&nproc);
    std::cout<<"my process rank: "<<rank<<", totoal process num: "<<nproc<<std::endl;
    const char *train_data_file = argv[1];
    const char *test_data_file = argv[2];

    long int glo_samp_num = 0;
    if(0 == rank) {
        std::ifstream tfin(train_data_file, std::ios::in);
        if(!tfin) std::cerr << "process "<< rank << " open error get feature number..." << train_data_file << std::endl;
        std::string line;
        while(getline(tfin, line)) {
                glo_samp_num++;
        }
        tfin.close();
    }
    MPI_Bcast(&glo_samp_num, 1, MPI_LONG, 0, MPI_COMM_WORLD);
    std::cout<< "process "<< rank <<", global sample number: "<<glo_samp_num<<std::endl;

    std::string split_tag = " ";
    Load_Data ld; 
    ld.load_data(train_data_file, split_tag, rank, nproc, glo_samp_num);
    std::cout<< "process "<< rank <<", local feature dimension: "<<ld.loc_fea_dim<<std::endl;
    std::cout<< "process "<< rank <<", local sample number: "<<ld.loc_samp_num<<std::endl;
    std::cout<< "process "<< rank <<", local sample number by matrix: "<<ld.fea_matrix.size()<<std::endl;
    LR lr(&ld, nproc, rank);
    //lr.run();
    MPI::Finalize();
    return 0;
}
