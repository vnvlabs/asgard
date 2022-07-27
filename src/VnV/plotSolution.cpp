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

/** @title PLOT SOLUTION Contour Plot Of the Solution 
   *
   * In this contour plot the x axis is the solution. The y 
   * axis is the time. So, this is a contour plot of the 1D 
   * solution against time. 
   *
   * .. vnv-plotly::
   *    :trace.main: contour
   *    :main.y: {{as_json(time)}}
   *    :main.z: {{as_json(solution)}}
   *    :layout.title.text: Asgard Solution against time.
   *    :layout.yaxis.title.text: time
   *    :layout.xaxis.title.text: index
   *
   * .. vnv-plotly::
   *    :trace.col: contour
   *    :trace.row: contour
   *    :col.x: {{as_json(nodes[-1])}}
   *    :col.y: {{as_json(nodes[-1])}}
   *    :col.z: {{as_json(solution_mat[-1])}}
   *    :row.x: {{as_json(nodes[0])}}
   *    :row.y: {{as_json(nodes[0])}}
   *    :row.z: {{as_json(solution_mat[0])}}
   *    :row.yaxis: y2
   *    :row.xaxis: x2
   *    :row.name: Initial
   *    :col.name: Final
   *    :layout.grid.rows: 1
   *    :layout.grid.columns: 2
   *    :layout.grid.pattern: independent
   *    :layout.title.text: Asgard Solution (t={{time[-1]}}) vs Initial Solution (t=0).
   *    :layout.yaxis.title.text: y
   *    :layout.xaxis.title.text: x
   *
   * .. vnv-plotly::
   *    :trace.col: scatter
   *    :trace.row: scatter
   *    :col.x: {{as_json(nodes[-1])}}
   *    :col.y: {{as_json(col[-1])}}
   *    :row.x: {{as_json(nodes[-1])}}
   *    :row.y: {{as_json(row[-1])}}
   *    :row.yaxis: y2
   *    :row.xaxis: x2
   *    :row.name: Horizontal
   *    :col.name: Vertical
   *    :layout.grid.rows: 1
   *    :layout.grid.columns: 2
   *    :layout.grid.pattern: independent
   *    :layout.title.text: 1D Solution Slices (t={{time[-1]}})
   *    :layout.xaxis.title.text: y
   *
   * .. vnv-plotly::
   *    :trace.col: contour
   *    :trace.row: contour
   *    :col.x: {{as_json(nodes[0])}}
   *    :col.y: {{as_json(time)}}
   *    :col.z: {{as_json(col)}}
   *    :row.x: {{as_json(nodes[0])}}
   *    :row.y: {{as_json(time)}}
   *    :row.z: {{as_json(row)}}
   *    :row.yaxis: y2
   *    :row.xaxis: x2
   *    :row.name: Horizontal
   *    :col.name: Vertical
   *    :layout.grid.rows: 1
   *    :layout.grid.columns: 2
   *    :layout.grid.pattern: independent
   *    :layout.title.text: 1D Solution Slices over time.
   *    :layout.yaxis.title.text: time
   *   
   **/ 
INJECTION_TEST(ASGARD, PlotSolution)
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

  // In this example I plot time vs. solution as a contour plot. 
  // We should just switch out the data with what we want.
  if (type == VnV::InjectionPointType::Begin) {}
  else if (type == VnV::InjectionPointType::Iter)
  {
    engine->Put("time", time); // Add the time value to the time vector;

    // Does fk::vector gauarantee contiguous storage? -- I guess we will find out.
    // This adds a new "vector" to the solution vector. After "X" timesteps we have

    // time --> 1x10 array of time values
    // solution 10x<GridSize> matrix of solution values.
    if (pde->num_dims > 1)
    {
      // get solution sizes for each dimension
      std::vector<int> sizes(pde->num_dims);
      for (int i = 0; i < pde->num_dims; i++)
      {
        sizes[i] = pde->get_dimensions()[i].get_degree() *
                   fm::two_raised_to(pde->get_dimensions()[i].get_level());
      }

      // fix this to be reused
      Solution<prec> *s = new Solution<prec>();

      // get the locations of each table element for each dimension
      auto dim                = pde->get_dimensions()[0];
      std::vector<prec> nodes = s->generate_nodes(
          dim.get_degree(), dim.get_level(), dim.domain_min, dim.domain_max);

      engine->Put_Vector("nodes", nodes.size(), nodes.data());

      // TODO: check if VnV supports plotly annotations? See below for attempt
      // at adding subplot titles:
      // :layout.annotations: [{ text: 'test', x: 0, y: 0, xref: 'paper', yref:
      // 'paper'}]

      // TODO: how to add X/Y axis labels to subplots? :layout.xaxis.title.text:
      // works for one, but not the other?

      // convert f_val from wavelet space to realspace for plotting
      static auto const default_workspace_cpu_MB = 187000;
      //auto const real_space_size                 = real_solution_size(*pde);
      auto const real_space_size = 1024;
      fk::vector<prec> real_space(real_space_size);

      // temporary workspaces for the transform
      // TODO: pull this out of iteration step, this can be reused each
      // iteration.
      fk::vector<prec, mem_type::owner, resource::host> workspace(
          real_space_size * 2);
      std::array<fk::vector<prec, mem_type::view, resource::host>, 2>
          tmp_workspace = {
              fk::vector<prec, mem_type::view, resource::host>(workspace, 0,
                                                               real_space_size),
              fk::vector<prec, mem_type::view, resource::host>(
                  workspace, real_space_size, real_space_size * 2 - 1)};
      // transform initial condition to realspace
      wavelet_to_realspace<prec>(*pde, f_val, adaptive_grid.get_table(),
                                 transformer, default_workspace_cpu_MB,
                                 tmp_workspace, real_space);

      // convert from fk::vector -> std::vector for vnv
      // auto sol = real_space.to_std();

      // TODO: this duplicates the solution but passes as a matrix to make the
      // contour plots. using the "vector" form does not seem to make the same
      // plot
      engine->Put_Matrix("solution_mat", sizes[0], sizes[1], real_space.data(), sizes[1]);
      
      // std::vector<prec> custom = real_space.to_std();
      // while (custom.size() != 1024) {
      //  if (custom.size() < 1024) {
      //          custom.push_back(0.0);
      //  }
      //  else {
      //          custom.pop_back();
      //  }
      //}
      //engine->Put_Vector("solution"+std::to_string(time), custom);
      engine->Put_Vector("solution"+std::to_string(time), sizes[0]*sizes[1], real_space.data());
      // for plotting 1d slices
      // TODO: should this reuse the above solution without saving more data to
      // the report?

       // interpret the real space solution as a matrix
       auto mat = fk::matrix<prec, mem_type::view, resource::host>(
           real_space, sizes[0], sizes[1]);

       // get indices for 1D horizontal/vertical slices
       int col = std::max(1, (int)std::floor((mat.ncols() / 2)) + 2);
       int row = std::max(1, (int)std::floor((mat.nrows() / 2)) + 2);

       auto row_slice = mat.extract_submatrix(row, 0, 1, mat.ncols());
       auto col_slice = mat.extract_submatrix(0, col, mat.nrows(), 1);

       engine->Put_Vector("row"+std::to_string(time), row_slice.size(), row_slice.data());
       engine->Put_Vector("col"+std::to_string(time), col_slice.size(), col_slice.data());
    }
    else
    {
      engine->Put_Vector("solution", f_val.size(), f_val.data());
    }
  }
  return SUCCESS;
}

#endif
