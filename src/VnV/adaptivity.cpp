#ifndef _VnV_adaptivity
#define _VnV_adaptivity

// #include "adapt.hpp"
#include "VnV.h"

INJECTION_REGISTRATION(MeshWatcher);

/**
 * The grid size is :vnv:`gridSize[0]` and maybe :vnv:`gridSize[1]`.
 **/
INJECTION_TEST(MeshWatcher, adaptivityTest, int64_t gridSize) {
  //adapt::distributed_grid<P> grid = get<adapt::distributed_grid<P>>("grid");
  const int64_t& gridSize = get<int64_t>("gridSize");
  engine->Put("gridSize", gridSize);
  return SUCCESS;
}

#endif
