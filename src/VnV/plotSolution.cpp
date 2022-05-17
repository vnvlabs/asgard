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


template<typename T>
class Solution
{
  std::vector<std::vector<T>> nodes_;
  std::vector<size_t> sol_sizes_;
  std::vector<std::vector<T>> elem_coords_;

  /* The next few functions are adapted from matlab_plot.cpp. */

public:
  std::vector<T>
  generate_nodes(int const degree, int const level, T const min, T const max)
  {
    // Trimmed version of matrix_plot_D.m to get only the nodes
    int const n        = pow(2, level);
    int const mat_dims = degree * n;
    T const h          = (max - min) / n;

    // TODO: fully implement the output_grid options from matlab (this is just
    // the 'else' case)
    auto const lgwt  = legendre_weights(degree, -1.0, 1.0, true);
    auto const roots = lgwt[0];

    unsigned int const dof = roots.size();

    auto nodes = std::vector<T>(mat_dims);

    for (int i = 0; i < n; i++)
    {
      auto p_val = legendre(roots, degree, legendre_normalization::lin);

      p_val[0] = p_val[0] * sqrt(1.0 / h);

      std::vector<T> xi(dof);
      for (std::size_t j = 0; j < dof; j++)
      {
        xi[j] = (0.5 * (roots(j) + 1.0) + i) * h + min;
      }

      std::vector<int> Iu(degree);
      for (int j = 0, je = degree - 1; j < je; j++)
      {
        Iu[j] = dof * i + j + 1;
      }
      Iu[degree - 1] = dof * (i + 1);

      for (std::size_t j = 0; j < dof; j++)
      {
        nodes[Iu[j] - 1] = xi[j];
      }
    }

    return nodes;
  }

  std::vector<std::vector<T>>
  gen_elem_coords(PDE<T> const &pde, elements::table const &table)
  {
    int const ndims = pde.num_dims;

    auto elem_coords = std::vector<std::vector<T>>(ndims);

    // Iterate over dimensions first since matlab needs col-major order
    for (int d = 0; d < ndims; d++)
    {
      elem_coords[d] = std::vector<T>(table.size());

      T const max = pde.get_dimensions()[d].domain_max;
      T const min = pde.get_dimensions()[d].domain_min;
      T const rng = max - min;

      for (int i = 0; i < table.size(); i++)
      {
        fk::vector<int> const &coords = table.get_coords(i);

        int const lev = coords(d);
        int const pos = coords(ndims + d);

        expect(lev >= 0);
        expect(pos >= 0);

        T x0;
        if (lev > 1)
        {
          T const s = pow(2, lev - 1) - 1.0;
          T const h = 1.0 / (pow(2, lev - 1));
          T const w = 1.0 - h;
          T const o = 0.5 * h;

          x0 = pos / s * w + o;
        }
        else
        {
          x0 = pos + 0.5;
        }

        T const x = x0 * rng + min;

        elem_coords[d][i] = x;
      }
    }

    return elem_coords;
  }

  inline std::vector<size_t> get_soln_sizes(PDE<T> const &pde)
  {
    // Returns a vector of the solution size for each dimension
    std::vector<size_t> sizes(pde.num_dims);
    for (int i = 0; i < pde.num_dims; i++)
    {
      sizes[i] = pde.get_dimensions()[i].get_degree() *
                 std::pow(2, pde.get_dimensions()[i].get_level());
    }
    return sizes;
  }

  void init_plotting(PDE<T> const &pde, elements::table const &table)
  {
    VnV_Info(ASGARD, "Initializing plotting");

    // Generates cell array of nodes and element coordinates needed for plotting
    sol_sizes_ = get_soln_sizes(pde);

    nodes_          = std::vector<std::vector<T>>(pde.num_dims);
    auto const dims = pde.get_dimensions();

    for (int i = 0; i < pde.num_dims; i++)
    {
      auto const &dim       = dims[i];
      auto const &node_list = generate_nodes(dim.get_degree(), dim.get_level(),
                                             dim.domain_min, dim.domain_max);
      nodes_[i]             = node_list;
    }

    elem_coords_ = gen_elem_coords(pde, table);

    VnV_Info(ASGARD, "Initialized plotting");
  }

  void plot_fval(PDE<T> const &pde, elements::table const &table,
                 fk::vector<T> const &f_val, fk::vector<T> const &analytic_soln)
  {
    expect(sol_sizes_.size() > 0);

    size_t const ndims = static_cast<size_t>(pde.num_dims);

    if (ndims != elem_coords_.size() || table.size() != elem_coords_[0].size())
    {
      // Regenerate the element coordinates and nodes if the grid was adapted
      init_plotting(pde, table);
    }

    /**
     * Now to plot using:
     *
     * analytic_soln
     * ndims
     * nodes_
     * sol_sizes_
     * f_val
     * elem_coords_
     */
  }

  std::map<std::string, int> count;

  std::string getStageId(std::string stage)
  {
    auto a = count.find(stage);
    if (a == count.end())
    {
      count[stage] = 0;
      return stage;
    }
    int c = a->second++;
    return stage + "_" + std::to_string(c);
  }
};

/** @title Contour Plot Of the Solution 
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
      auto const real_space_size                 = real_solution_size(*pde);
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
      engine->Put_Matrix("solution_mat", sizes[0], sizes[1], real_space.data(),
                         sizes[1]);
      engine->Put_Vector("solution", sizes[0] * sizes[1], real_space.data());

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
