#include <string>
#include "load_data.h"
#include "ftrl.h"
#include "predict.h"

void StartServer(){
    if(!ps::IsServer()) return;
    auto server = new ps::KVServer<float>(0);
    server->set_request_handle(ps::KVServerDefaultHandle<float>());
    ps::RegisterExitCallback([server]()){
	delete server;
    };
}

int main(int argc,char* argv[]){  
    StartServer();
    ps::Start(); 
    if(ps::IsWorker()){
	int rank = ps::MyRank();
        char train_data_path[1024];
        const char *train_data_file = argv[2];
        snprintf(train_data_path, 1024, "%s-%05d", train_data_file, rank);

        Load_Data load_data; 
        load_data.load_data(train_data_path);
        //std::cout<<ld.fea_matrix.size()<<std::endl;    
        FTRL ftrl(&load_data);
        ftrl.run();

    ps::Finalize();
    return 0;
}
