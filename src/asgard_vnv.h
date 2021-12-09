#ifdef ASGARD_USE_VNV
#include "VnV.h"
#else
  #define INJECTION_INITIALIZE(...) 
  #define INJECTION_FINALIZE(...)
  #define INJECTION_LOOP_BEGIN(...)
  #define INJECTION_LOOP_END(...)
  #define INJECTION_LOOP_ITER(...)   
#endif