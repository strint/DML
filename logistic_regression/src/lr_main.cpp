#include "lr.h"
#include "load_data.h"
#include "mpi.h"
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <iomanip>
#include "gtest/gtest.h"
#include "config.h"

struct ThreadParam{
    LR *lr;
    int threads_num;
    int process_id;
    int n_process;
};

void *opt_algo(void *arg){
   ThreadParam *args = (ThreadParam*)arg;
   args->lr->owlqn(args->process_id, args->n_process); 
}

int main(int argc,char* argv[]){  
    std::cout<<"cmd: ./train -trainfile -testfile"<<std::endl;
    int myid, numprocs;
    MPI_Status status;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&myid);
    MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
    //exec by main thread    
    const char *train_data_file = argv[1];
    char *test_data_file = argv[2];
    std::string split_tag = " ";
    
    Load_Data ld;
    ld.fea_dim = 0;
    ld.load_data(train_data_file, split_tag);
    int root = 0;
    MPI_Bcast(&ld.fea_dim, 1, MPI_INT, root, MPI_COMM_WORLD);
    std::cout<<ld.fea_dim<<std::endl;
    std::vector<ThreadParam> params;
    std::vector<pthread_t> threads;
    CONFIG config;
    config.n_threads = 2;
    for(int i = 0; i < config.n_threads; i++){//construct parameter
        LR lr;
        lr.init_theta();
        ThreadParam param = {&lr, config.n_threads, myid, numprocs};
        params.push_back(param);
    } 
    //multithread start
    for(int i = 0; i < params.size(); i++){
        pthread_t thread_id;
        std::cout<<thread_id<<std::endl;
        int ret = pthread_create(&thread_id, NULL, &opt_algo, (void*)&(params[i])); 
        if(ret != 0) std::cout<<"process "<<i<<"failed(create thread faild.)"<<std::endl;
        else threads.push_back(thread_id);
    }
    for(int i = 0; i < threads.size(); i++){//join threads function
        pthread_join(threads[i], 0); 
    }
   
    MPI::Finalize();
    return 0;
}
