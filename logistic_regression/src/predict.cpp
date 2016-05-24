#include <iostream>
#include <fstream>
#include "predict.h"

Predict::Predict(){}
Predict::~Predict(){}

std::vector<std::string> Predict::split_line(const std::string& line) {
    const std::string split_tag = "\t";
    size_t start = 0, end = 0;
    std::vector<std::string> feature_index;
    feature_index.clear();
    while((end = line.find_first_of(split_tag, start)) != std::string::npos) {
        if(end > start){
            feature_index.push_back(line.substr(start, end - start));
        }
        start = end + 1;
    }
    if(start < line.size()){
        feature_index.push_back(line.substr(start));
    }
}

void Predict::predict(const char* test_file,  std::vector<float>& theta){
    std::cout<<"predic start-----------------------------------"<<std::endl;
    std::ifstream fin(test_file);
    std::string test_line;
    std::vector<float> predict_result;
    std::vector<std::string> predict_feature;
    float x;
    std::vector<int> preindex;
    std::vector<float> preval;
    while(getline(fin, test_line)){
        x = 0.0;
        predict_feature.clear();
        predict_feature = split_line(test_line);
        preindex.clear();
        preval.clear();
        for(size_t j = 0; j < predict_feature.size(); j++){
            int beg = 0, end = 0;
            while((end = predict_feature[j].find_first_of(":",beg)) != std::string::npos){
                if(end > beg){
                    std::string string_sub = predict_feature[j].substr(beg, end - beg);
                    int k = atoi(string_sub.c_str());
                    preindex.push_back(k-1);
                }
                beg = end + 1;
            }
            std::string string_end = predict_feature[j].substr(beg);
            int t = atoi(string_end.c_str());
            preval.push_back(t);
        }
        float y = 0.0;
        for(size_t j = 0; j < preindex.size(); j++){
            x += theta[preindex[j]] * preval[j];
        }
        if(x < -30){
            y = 1e-6;
        }
        else if(x > 30){
            y = 1.0;
        }
        else{
            double ex = pow(2.718281828, x);
            y = ex / (1.0 + ex);
        }
        predict_result.push_back(y);
    }
    for(size_t j = 0; j < predict_result.size(); j++){
        std::cout<<predict_result[j]<<std::endl;
    }
}

