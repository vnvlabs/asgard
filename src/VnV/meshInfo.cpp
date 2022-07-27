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

/** @title Mesh Information 
   *
   * The following displays information about the adaptive mesh. 
   *
   * .. vnv-plotly::
   *    :trace.degree: scatter
   *    :trace.level: scatter
   *    :trace.size: scatter
   *    :trace.num_elem: scatter
   *    :degree.x: {{as_json(time)}}
   *    :degree.y: {{as_json(degree)}}
   *    :level.x: {{as_json(time)}}
   *    :level.y: {{as_json(level)}}
   *    :size.x: {{as_json(time)}}
   *    :size.y: {{as_json(size)}}
   *    :num_elem.x: {{as_json(time)}}
   *    :num_elem.y: {{as_json(num_elem)}}
   *    :layout.title.text: Mesh Information vs. Time
   *    :layout.yaxis.title.text: Number
   *    :layout.xaxis.title.text: Time
   *
**/ 
INJECTION_TEST(ASGARD, MeshInfo)
{
  // Can't use T for type parameter in these two GetRef conversions.
  auto &adaptive_grid = GetRef_NoCheck("adaptive_grid", adapt::distributed_grid<prec>);
  auto &pde  = GetRef_NoCheck("pde", std::unique_ptr<PDE<prec>>);
  auto &time = GetRef_NoCheck("time", prec);
  
  if (type == VnV::InjectionPointType::Begin) 
  {
  }
  else if (type == VnV::InjectionPointType::Iter)
  {
    engine->Put("time", time);

    //TODO expand to multiple dims
    auto dims_0 = pde->get_dimensions()[0];
    auto deg_0 = dims_0.get_degree();
    auto lev_0 = dims_0.get_level();
    engine->Put("degree", deg_0);
    engine->Put("level", lev_0);
    engine->Put("size", deg_0 * fm::two_raised_to(lev_0));
    engine->Put("num_elem", adaptive_grid.size());

    return SUCCESS;
  }
}
#endif
