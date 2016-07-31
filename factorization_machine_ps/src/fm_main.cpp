#include <string>
#include "load_data.h"
#include "ftrl.h"
#include "ps/ps.h"

void StartServer(){
    if(!ps::IsServer()) return;
    auto server = new ps::KVServer<float>(0);
    server->set_request_handle(ps::KVServerDefaultHandle<float>());
    ps::RegisterExitCallback([server](){ delete server; });
}

int main(int argc,char* argv[]){  
    StartServer();
    ps::Start();
    int rank = ps::MyRank();
    if(ps::IsWorker()){
        char train_data_path[1024];
        const char *train_data_file = argv[1];
        snprintf(train_data_path, 1024, "%s-%05d", train_data_file, rank);
        Load_Data load_data(train_data_path); 
        FTRL ftrl(&load_data);
        ftrl.run();
    }
    ps::Finalize();
    return 0;
}
