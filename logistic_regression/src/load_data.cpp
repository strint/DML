#include <iostream>
#include <fstream>
#include "load_data.h"
#include "stdlib.h"

Load_Data::Load_Data(){}
Load_Data::~Load_Data(){}

std::vector<std::string> Load_Data::split_line(std::string split_tag, std::vector<std::string>& feature_index){
    int start = 0, end = 0;
    while((end = line.find_first_of(split_tag, start)) != std::string::npos){
        if(end > start){
            index_str = line.substr(start, end - start);
            feature_index.push_back(index_str);
        }
        start = end + 1;
    }
    if(start < line.size()){
        index_str = line.substr(start);
        feature_index.push_back(index_str);
    }
    std::cout<<feature_index.size()<<std::endl;
}

void Load_Data::get_feature_struct(){
    std::cout<<feature_index.size()<<std::endl;
    for(int i = 1; i < feature_index.size(); i++){//start from index 1
        std::cout<<feature_index[i]<<std::endl; 
        int start = 0, end = 0;
        while((end = feature_index[i].find_first_of(":", start)) != std::string::npos){
            if(end > start){
                index_str = feature_index[i].substr(start, end - start);
                float index = atoi(index_str.c_str());
                if(index > fea_dim) fea_dim = index + 1;
                sf.idx = index - 1;
            }
            //beg += 1; //this code must remain,it makes me crazy two days!!!
            start = end + 1;
        }
        if(start < feature_index[i].size()){
            index_str = feature_index[i].substr(start);
            float value = atoi(index_str.c_str());
            sf.val = value;
        }
        key_val.push_back(sf);
    }
}

void Load_Data::load_data(const char* data_file, std::string split_tag){
    std::ifstream fin(data_file, std::ios::in);
    if(!fin) std::cerr<<"open error get feature number..."<<data_file<<std::endl;
    int y = 0;
    while(getline(fin,line)){
        std::cout<<line<<std::endl;
        feature_index.clear();
        key_val.clear();
        //return id:value, .e.g 3:1, 4:1
        std::cout<<feature_index.size()<<std::endl;
        split_line(split_tag, feature_index);
        std::cout<<feature_index.size()<<std::endl;
        //y = atof(feature_index[0].c_str());
        //label.push_back(y);
        //get_feature_struct();
        //fea_matrix.push_back(key_val);
    }
    fin.close();
}
