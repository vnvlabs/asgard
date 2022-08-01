#ifndef _VnV_adaptivity
#define _VnV_adaptivity

#include "../adapt.hpp"
#include "VnV.h"
#include "../program_options.hpp"
#include "../transformations.hpp"

#ifdef ASGARD_USE_DOUBLE_PREC
using prec = double;
#else
using prec = float;
#endif

#include "./solution.hpp"

/*** @title Coordinates 
   *
   * Displays coordinates.
   *
   * .. vnv-plotly::
   *    :trace.degree: scatter
   *    :degree.x: {{as_json(elem_coords0)}}
   *    :degree.y: {{as_json(elem_coords1)}}
   *
***/ 
INJECTION_TEST(ASGARD, PlotCoordinates)
{
  // Can't use T for type parameter in these two GetRef conversions.
  auto &adaptive_grid = GetRef_NoCheck("adaptive_grid", adapt::distributed_grid<prec>);
  auto &pde  = GetRef_NoCheck("pde", std::unique_ptr<PDE<prec>>);
  auto &time = GetRef_NoCheck("time", prec);
  auto &opts = GetRef_NoCheck("opts", options);
  
  if (type == VnV::InjectionPointType::Begin) 
  {
  }
  else if (type == VnV::InjectionPointType::Iter)
  {
    engine->Put("time", time);
    
    elements::table table(opts, *pde);

    Solution<prec> *s = new Solution<prec>();
    s->init_plotting(*pde, table);
    s->put_to_vnv(*pde);
    //nodes is a std::vector<prec>(degree*pow(2,level)=mat_dims)
    //std::vector<std::vector<prec>> nodes = s->nodes_;
    //std::vector<std::vector<prec>> elem_coords = s->elem_coords_;
    //std::vector<std::vector<prec>> elem_coords_ = gen_elem_coords(*pde, table);
    
    //engine->Put_Vector("x", elem_coords.at(0));
    //engine->Put_Vector("y", elem_coords.at(1));

    return SUCCESS;
  }
}
#endif

