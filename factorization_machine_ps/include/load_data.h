#ifndef LOAD_DATA_H_
#define LOAD_DATA_H_

#include <fstream>
#include <iostream>
#include <vector>

struct sparse_feature{
    long int idx;
    int val;
};

class Load_Data {
private:
    std::ifstream fin_;
    std::vector<std::vector<sparse_feature> > fea_matrix;
    std::vector<sparse_feature> key_val;
    sparse_feature sf;
    std::vector<int> label;
    std::string line;
    int y, value, nchar;
    long int index;

public:
    Load_Data(const char * file_name){
	fin_.open(file_name, std::ios::in);
    }

    load_data_minibatch(const int num){
        fea_matrix.clear();
    	for(int i = 0; i < num; i++){
	    if(fin_.eof()) break;
	    std::getline(fin_, line);
            key_val.clear();
            const char *pline = line.c_str();
            if(sscanf(pline, "%d%n", &y, &nchar) >= 1){
                pline += nchar;
                label.push_back(y);
                while(sscanf(pline, "%ld:%d%n", &index, &value, &nchar) >= 2){
                    pline += nchar;
                    sf.idx = index;
                    sf.val = value;
                    key_val.push_back(sf);
                }
            }
            fea_matrix.push_back(key_val);
	}
	return fea_matrix;
    }

    ~Load_Data(){
        fin_.close();
    };
};
#endif
