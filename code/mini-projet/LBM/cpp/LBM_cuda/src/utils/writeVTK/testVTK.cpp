#include <cstdlib>
#include <iostream>

#include "utils/writeVTK/saveVTK.h"
#include "utils/config/ConfigMap.h"

int main(int argc, char* argv[])
{

  std::cout << "Testing routine saveVTK....\n";

  // read parameter file and initialize parameter
  // parse parameters from input file
  std::string input_file = argc>1 ? std::string(argv[1]) : "flowAroundCylinder.ini";

  ConfigMap configMap(input_file);

  // test: create a LBMParams object
  LBMParams params = LBMParams();
  params.setup(configMap);

  // print parameters on screen
  params.print();

  const int nx = params.nx;
  const int ny = params.ny;

  real_t* rho = (real_t*) calloc(nx*ny,sizeof(real_t));
  real_t* ux  = (real_t*) calloc(nx*ny,sizeof(real_t));
  real_t* uy  = (real_t*) calloc(nx*ny,sizeof(real_t));

  // create fake data
  {
    for (int j=0; j<ny; ++j)
      for (int i=0; i<nx; ++i) {
        int index = i+nx*j;
        rho[index] = 2.0*i+j;
        ux[index] = 1.0*i*i-j;
        uy[index] = 1.0*j*j-i;
      }
  }

  bool useAscii = true;
  saveVTK(rho, ux, uy, params, useAscii, 0);

  // free memory
  delete[] rho;
  delete[] ux;
  delete[] uy;

  return EXIT_SUCCESS;
}
