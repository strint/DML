#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <string>
#include <map>
#include <set>
#include <vector>
#include <string.h>

#define CTRL_A "	"
#define CTRL_B ""

#define MASTER_ID (0)
#define MPI_NON_CLK_TAG (0)
#define MPI_CLK_TAG (1)

typedef struct{
        float clk;
        float nclk;
        long idx;
} clkinfo;

class Load_Data{
public:
    Load_Data();
    ~Load_Data();
    int load_pctr_nclk_clk(const char* p_str_ins_path, int rank);
    std::vector<clkinfo> clkinfo_list;
    int MAX_ARRAY_SIZE;
private:
    void init();
    float pctr;
    float nclk;
    float clk;
};
