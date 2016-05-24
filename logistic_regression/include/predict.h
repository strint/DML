#ifndef PREDICT_H_
#define PREDICT_H_

#include <string>
#include <vector>
#include <math.h>
class Predict{

public:
    Predict();
    ~Predict();

    void predict(const char*, std::vector<float>&);
    std::vector<std::string> split_line(const std::string&);
     

};
#endif
