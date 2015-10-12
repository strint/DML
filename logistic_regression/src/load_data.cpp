#include "loan_data.h"

std::vector<std::string> OPT_ALGO::split_line(std::string split_tag, std::vector<std::string>& feature_index){
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
}

void OPT_ALGO::get_feature_struct(){
    for(int i = 1; i < feature_index.size(); i++){//start from index 1
        int start = 0, end = 0;
        while((end = feature_index[i].find_first_of(":", start)) != std::string::npos){
            if(end > start){
                index_str = feature_index[i].substr(start, end - start);
                index = atoi(index_str.c_str());
                if(index > fea_dim) fea_dim = index + 1;
                sf.idx = index - 1;
            }
            //beg += 1; //this code must remain,it makes me crazy two days!!!
            start = end + 1;
        }
        if(start < feature_index[i].size()){
            index_str = feature_index[i].substr(start);
            value = atoi(index_str.c_str());
            sf.val = value;
        }
        key_val.push_back(sf);
    }
}

void OPT_ALGO::load_data(std::string data_file, std::string split_tag){
    std::ifstream fin(data_file.c_str(), std::ios::in);
    if(!fin) std::cerr<<"open error get feature number..."<<data_file<<std::endl;
    int y = 0;
    while(getline(fin,line)){
        feature_index.clear();
        key_val.clear();
        //return id:value, .e.g 3:1, 4:1
        split_line(split_tag, feature_index);
        y = atof(feature_index[0].c_str());
        label.push_back(y);
        //3:1 as input
        get_feature_struct();
        fea_matrix.push_back(key_val);
    }
    fin.close();
}
