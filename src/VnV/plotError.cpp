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

/**
 * @title Plots of Solution Error
 *
 * -----------------------------------------
 * 
 * ===============================================================
 * Scatter Plots of the Number of Elements and Time Steps Over Time
 * ===============================================================
 *
 * ::
 *
 *      How to read these data:
 *              TODO
 *
 * .. vnv-plotly::
 *     :trace.time: scatter
 *     :trace.elms: scatter
 *     :dt.y: {{time}}
 *     :elms.y: {{elements}}
 *     :elms.yaxis: y2
 *     :elms.xaxis: x2
 *     :layout.grid.rows: 1
 *     :layout.grid.columns: 2
 *     :layout.grid.pattern: independent
 *     :layout.title.text: Elements and Time
 *
 * -----------------------------------------
 *
 * .. vnv-if:: analytic
 *   
 *   ======================================
 *   Scatter Plot of Root Mean Square Error
 *   ======================================
 *
 *   ::
 *
 *      How to read these data:
 *              TODO
 *
 *   .. vnv-plotly::
 *       :trace.rmse: scatter
 *       :trace.rel: scatter
 *       :rmse.y: {{rmse}}
 *       :rel.y: {{rel}}
 *       :rel.yaxis: y2
 *       :rel.xaxis: x2
 *       :layout.grid.rows: 1
 *       :layout.grid.columns: 2
 *       :layout.grid.pattern: independent
 *       :layout.title.text: Root Mean Square Error
 *
 *       -----------------------------------------
 *
 **/
INJECTION_TEST(ASGARD, PlotError)
{
  // Can't use T for type parameter in these two GetRef conversions.
  auto &adaptive_grid = GetRef_NoCheck("adaptive_grid", adapt::distributed_grid<prec>);
  auto &pde  = GetRef_NoCheck("pde", std::unique_ptr<PDE<prec>>);
  auto &opts = GetRef_NoCheck("opts", options);
  auto &time = GetRef_NoCheck("time", prec);
  auto &f_val = GetRef_NoCheck("f_val", fk::vector<prec>);
  
  // Bug -- GetRef_NoCheck does not support types with a "," in them :?
  void *transformer_raw = GetPtr_NoCheck("transformer", void );
  auto* transformer_ptr = (basis::wavelet_transform<prec, resource::host>*) transformer_raw;
  auto& transformer = *transformer_ptr;


  auto const degree = pde->get_dimensions()[0].get_degree();

  engine->Put("elements", adaptive_grid.size());

  if (type == VnV::InjectionPointType::Begin)
  {
    engine->Put("nts", opts.num_time_steps);
    engine->Put("analytic", pde->has_analytic_soln);
  }
  else if (type == VnV::InjectionPointType::Iter)
  {
    engine->Put("time", time);

    // print root mean squared error from analytic solution
    if (pde->has_analytic_soln)
    {
      // get analytic solution at time(step+1)
      auto const subgrid           = adaptive_grid.get_subgrid(get_rank());
      auto const time_multiplier   = pde->exact_time(time + pde->get_dt());
      auto const analytic_solution = transform_and_combine_dimensions(
          *pde, pde->exact_vector_funcs, adaptive_grid.get_table(), transformer,
          subgrid.col_start, subgrid.col_stop, degree, time, time_multiplier);

      // calculate root mean squared error
      auto const diff = f_val - analytic_solution;
      auto const RMSE = [&diff]() {
        fk::vector<prec> squared(diff);
        std::transform(squared.begin(), squared.end(), squared.begin(),
                       [](prec const &elem) { return elem * elem; });
        auto const mean = std::accumulate(squared.begin(), squared.end(), 0.0) /
                          squared.size();
        return std::sqrt(mean);
      }();

      // Only writes rank 0 result for now -- Change to Put_Rank to
      // write values for
      // all ranks into a rank indexed vector.
      engine->Put("rmse", RMSE);
      engine->Put("rel", RMSE / inf_norm(analytic_solution) * 100);
    }
  }
  return SUCCESS;
}

#endif

