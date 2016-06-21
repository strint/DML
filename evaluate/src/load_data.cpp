#include <math.h>
#include "load_data.h"

Load_Data::Load_Data(){
    init();
}

Load_Data::~Load_Data(){}

void Load_Data::init(){
    pctr = 0.0;
    nclk = 0;
    clk = 0;
    MAX_ARRAY_SIZE = 1000;
}

int Load_Data::load_pctr_nclk_clk(const char* str_ins_path, int rank){
    std::ifstream ifs;
    std::string line = "";
    std::string tmpstr = "";
    std::string qid = "";
    char ins_path[2048];
 
    snprintf(ins_path, 2048, "%s-%05d", str_ins_path, rank);
    
    ifs.open(ins_path);
    clkinfo_list.clear();
    if (!ifs.is_open()) {
        std::cout<<ins_path<<std::endl;
    }

    while(getline(ifs, line)){
        int pos = line.find(CTRL_B);
        if(pos <= 0) tmpstr = line;
        else tmpstr = line.substr(0, pos);
        
        pos = tmpstr.find(CTRL_A);
        pctr = atof(tmpstr.substr(0, pos).c_str());
        tmpstr = tmpstr.substr(pos+1, tmpstr.size() - pos -1);
        //std::cout<<"pctr = "<<pctr<<" ";

        pos = tmpstr.find(CTRL_A);
        nclk = atof(tmpstr.substr(0, pos).c_str());
        tmpstr = tmpstr.substr(pos+1, tmpstr.size() - pos -1);
        //std::cout<<"nclk = "<<nclk<<" ";
        pos = tmpstr.find(CTRL_B);
        clk = atof(tmpstr.substr(0, pos).c_str());  
        //std::cout<<"clk = "<<clk<<std::endl;        

        int id = int(pctr*MAX_ARRAY_SIZE);
        
        clkinfo _clkinfo;
        _clkinfo.nclk = nclk;
        _clkinfo.clk = clk;
        _clkinfo.idx = id;
        clkinfo_list.push_back(_clkinfo);
    }
    //std::cout<<clkinfo_list.size()<<std::endl;
}
