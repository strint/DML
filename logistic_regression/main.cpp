#include <cstring>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include "gtest/gtest.h"
#include "opt_algo.h"
#include "mpi.h"

struct ThreadParam{
    OPT_ALGO *opt;
    pid_t main_thread_id;
    int process_id;
    int n_process;
};

void *opt_algo(void* arg){
   ThreadParam *args = (ThreadParam*)arg;
   args->opt->owlqn(args->process_id, args->n_process); 
   return NULL;
}

int main(int argc,char* argv[]){
    int myid, numprocs;
    // MPI_Status status;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&myid);
    MPI_Comm_size(MPI_COMM_WORLD,&numprocs);

    std::string train_data_file("./data/traindata.txt");
    std::string test_data_file("./data/testdata.txt");
    std::string split_tag = "\t";

    //call by main process
    OPT_ALGO opt(train_data_file, split_tag);
    MPI_Bcast(&opt.fea_dim, 1, MPI_INT, 0, MPI_COMM_WORLD);

    //multithread start
    std::vector<ThreadParam> params;
    std::vector<pthread_t> threads;
    for(int i = 0; i < opt.n_threads; i++) {//construct parameter
        ThreadParam param = {&opt, myid, numprocs};
        params.push_back(param);
    } 
    // pid_t main_thread_id;
    // main_thread_id = getpid();
    for(size_t i = 0; i < params.size(); i++){
        pthread_t thread_id;
        int ret = pthread_create(&thread_id, NULL, opt_algo, (void*)&(params[i])); 
        if(ret != 0) std::cout<<"process "<<i<<"failed(create thread faild.)"<<std::endl;
        else threads.push_back(thread_id);
    }
    for(size_t i = 0; i < threads.size(); i++){//join threads function
        pthread_join(threads[i], 0);
    }

    MPI::Finalize();
    return 0;
}
