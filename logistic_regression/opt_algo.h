#include <string>
#include <fstream>
#include <vector>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <deque>
using namespace std;

struct sparse_feature{
    int idx;
    float val;  
};
class OPT_ALGO{
public:
    OPT_ALGO();
    ~OPT_ALGO();
    vector<vector<sparse_feature> > feature_matrix;//feature matrix
    vector<int> label;
    vector<float> w;//parameter of logistic regression
    
    void load_one_sample(string sample_file);  
    float sigmoid(float x);
    void sgd(vector<float> &w, int myid, int numprocs);

    void owlqn(vector<float> w, int myid, int numprocs);
    float fun(vector<float>& w);
    void grad(vector<float>& w, vector<float>& g);
    void sub_gradient(vector<float>& w, vector<float>& g, vector<float>& sub_g);
    void two_loop(vector<float>& sub_g);
    void parallel_owlqn(float old_val, float newval);
    void fixdir(vector<float>& sub_g, vector<float>& g);
    void linesearch(float old_f, vector<float>& w, vector<float>&sub_g, vector<float>& g, vector<float>& nextw);
private:
    string filename;
    deque<vector<float> > ylist;
    deque<vector<float> > slist;
    vector<float> rolist;
    vector<float> g, sub_g;
    float f, nextw;
    float c;
    int dim, m;   
};