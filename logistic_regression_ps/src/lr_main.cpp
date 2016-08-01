#include "worker.h"
#include "server.h"
#include "scheduler.h"

#include "ps.h"

namespace ps{
  App* App::Create(int argc, char *argv[]){
    int rank = ps::MyRank();
    char train_data_path[1024];
    const char *train_data_file = argv[1];
    snprintf(train_data_path, 1024, "%s-%05d", train_data_file, rank);

    NodeInfo n;
    if(n.IsWorker()){
	std::cout<<"create worker~"<<std::endl;
        return new ::dmlc::linear::Worker(train_data_path);
    }else if(n.IsServer()){
	std::cout<<"create server~"<<std::endl;
        return new ::dmlc::linear::Server();
    }else if(n.IsScheduler){
	std::cout<<"create scheduler~"<<std::endl;
        return new ::dmlc::linear::Scheduler();
    }
    return NULL;
  }

}//namespace ps

int64_t dmlc::linear::ISGDHandle::new_w = 0;

int main(int argc,char *argv[]){  
    return ps::RunSystem(&argc, &argv);
}
