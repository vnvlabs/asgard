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

/** @title Plots of PDE Solution
   * 
   * -----------------------------------------
   *
   * =========================================
   * Countour plot of the 1D solution vs. time
   * =========================================
   * 
   * ::
   *
   *    How to read these data:
   *            This contour plot creates a row of solutions for each instance of time.
   *            The solution at the first instance of time is the bottomost row, 
   *            the subsequent solution at the next instance of time is the row 2nd from the bottom, and so on.
   *            This continues until the uppermost row is reached, where this row 
   *            corresponds to the solution of the PDE at the final instance of time.
   *            These rows are created by flattening the 2D solution into a 1D vector.
   *
   * .. vnv-plotly::
   *    :trace.main: contour
   *    :main.y: {{as_json(time)}}
   *    :main.z: {{as_json(solution)}}
   *    :layout.title.text: Asgard Solution against time.
   *    :layout.yaxis.title.text: time
   *    :layout.xaxis.title.text: solution index
   * 
   * -----------------------------------------
   *
   * ================================================
   * Snapshots of the contour plot of the 2D solution
   * ================================================
   * 
   * ..
   * 	TODO un-hardcode time values (i.e., replace t=0 with something like t={{time[0]}})
   *
   * ::
   * 
   *    How to read these data:
   *            These 2 plots display the solution to the 2D continuity equation at different instances of time.
   *            The plot on the left shows the solution at the end of the simulation.
   *            On the other hand, the plot on the right 
   *            shows the solution at the beginning of the simulation, which is at t=0.
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
   *
   * -----------------------------------------
   *
   * ===========================================
   * Countour plots of the 1D solutions vs. time 
   * ===========================================
   * 
   * ..
   * 	TODO un-hardcode (x,y) coordinates
   * 
   * ::
   * 
   *    How to read these data:
   *            These plots show how the solution located within a 1D line segment 
   *            from the full 2D solution changes over time.
   *            To help visualize this, imagine that you have a pencil and 
   *            want to draw a straight line across the solution this PDE at the initial point in time.
   *            The following two plots show two ways that you can draw a line with the pencil: 
   *            you can create a horizontal line that incorporates various columns on a single row, 
   *            or you can create a vertical line that incorporates various rows on a single column.
   *            These plots do exactly that, where the plot on the left shows how the solution 
   *            located at a line through columns starting at (x,y)=(-0.8943376,0.6056624) 
   *            and ending at (x,y)=(0.8943376,0.6056624) evolves over time.
   *            Likewise, the plot on the right shows how the solution 
   *            located at a line through rows starting at (x,y)=(0.6056624,-0.8943376) 
   *            and ending at (x,y)=(0.6056624,-0.8943376) evolves over time.
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
   * -----------------------------------------
   *
   * ======================================
   * Scatter plots of the final 1D solution
   * ======================================
   *
   * ::
   * 
   *    How to read these data:
   *            These plots are slices of the 1D Solutions vs. Time plots.
   *            The plot on the left is a slice of the solution incorporating 
   *            various columns on a single row at the first instance of time.
   *            Similarly, the plot on the right is a slice of the solution incorporating 
   *            various rows on a single column at the first instance of time.
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
   *
   * -----------------------------------------
   *
   * ======================================
   * Animation of the 2D solution over time
   * ======================================
   *
   * ::
   *
   *    How to read these data:
   *            This plot animates the PDE dynamics over time.
   *            Click the Play button to start the animation.
   *            Dragging the slider to the very left will set the time to the initial time, 
   *            and dragging it to the right will bring the simulation to a later time.
   *
   * ..
   * 	TODO remove numbers that are oddly displayed near the animation slider
   * 	TODO figure out where those numbers come from and what they mean
   *
   * .. vnv-animation::
   *    :trace.sol: contour
   *    :layout.title.text: 2D Solution
   *    :values: {{solution_mat}}
   *    :sol.x: {{nodes[0]}}
   *    :sol.y: {{nodes[0]}}
   *    :sol.z: ${i}
   *    :layout.xaxis.title.text: x
   *    :layout.yaxis.title.text: y
   *
   * -----------------------------------------
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
      engine->Put_Vector("solution", sizes[0]*sizes[1], real_space.data());
     
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

       engine->Put_Vector("row", row_slice.size(), row_slice.data());
       engine->Put_Vector("col", col_slice.size(), col_slice.data());
    }
    else
    {
      engine->Put_Vector("solution", f_val.size(), f_val.data());
    }
  }
  return SUCCESS;
}

#endif

