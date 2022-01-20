#ifndef _VnV_adaptivity
#define _VnV_adaptivity

#include "../adapt.hpp"
#include "VnV.h"

#ifdef ASGARD_USE_DOUBLE_PREC
using prec = double;
#else
using prec = float;
#endif

INJECTION_REGISTRATION(MeshWatcher);

/**
 * Asgard Adaptivity Tracking.
 * ===========================
 *
 * The final grid had :vnv:`Elements[-1]` elements.
 *
 * .. vnv-chart::
 *
 *    {
 *       "type" : "line",
 *       "data" : {
 *          "labels" : {{as_json(Labels)}},
 *          "datasets" : [{
 *             "label": "Number of elements",
 *             "backgroundColor": "rgb(255, 99, 132)",
 *             "borderColor": "rgb(255, 99, 132)",
 *             "data": {{as_json(Elements)}}
 *           }]
 *       },
 *       "options" : {
 *           "animation" : false,
 *           "responsive" : true,
 *           "title" : { "display" : true,
 *                       "text" : "The Number of elements in the adaptive grid."
 *                     },
 *          "scales": {
 *             "yAxes": [{
 *               "scaleLabel": {
 *                 "display": true,
 *                 "labelString": "Number of Elements"
 *               }
 *            }],
 *            "xAxes": [{
 *              "scaleLabel": {
 *                 "display":true,
 *                 "labelString": "Injection Point Stage"
 *               }
 *            }]
 *          }
 *       }
 *    }
 *
 *
 */

/* The next few functions are adapted from matlab_plot.cpp. */

std::vector<prec> generate_nodes(int const degree, int const level,
                                prec const min, prec const max)
{
  // Trimmed version of matrix_plot_D.m to get only the nodes
  int const n = pow(2, level);
  int const mat_dims = degree * n;
  prec const h = (max - min) / n;

  // TODO: fully implement the output_grid options from matlab (this is just
  // the 'else' case)
  auto const lgwt  = legendre_weights(degree, -1.0, 1.0, true);
  auto const roots = lgwt[0];

  unsigned int const dof = roots.size();

  auto nodes = std::vector<prec>(mat_dims);

  for (int i = 0; i < n; i++)
  {
    auto p_val = legendre(roots, degree, legendre_normalization::lin);

    p_val[0] = p_val[0] * sqrt(1.0 / h);

    std::vector<prec> xi(dof);
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

std::vector<std::vector<prec>> gen_elem_coords(PDE<prec> const &pde,
                                               elements::table const &table)
{
  int const ndims = pde.num_dims;

  auto elem_coords = std::vector<std::vector<prec>>(ndims);

  // Iterate over dimensions first since matlab needs col-major order
  for (int d = 0; d < ndims; d++)
  {
    elem_coords[d] = std::vector<prec>(table.size());

    prec const max = pde.get_dimensions()[d].domain_max;
    prec const min = pde.get_dimensions()[d].domain_min;
    prec const rng = max - min;

    for (int i = 0; i < table.size(); i++)
    {
      fk::vector<int> const &coords = table.get_coords(i);

      int const lev = coords(d);
      int const pos = coords(ndims + d);

      expect(lev >= 0);
      expect(pos >= 0);

      prec x0;
      if (lev > 1)
      {
        prec const s = pow(2, lev - 1) - 1.0;
        prec const h = 1.0 / (pow(2, lev - 1));
        prec const w = 1.0 - h;
        prec const o = 0.5 * h;

        x0 = pos / s * w + o;
      }
      else
      {
        x0 = pos + 0.5;
      }

      prec const x = x0 * rng + min;

      elem_coords[d][i] = x;
    }
  }

  return elem_coords;
}

inline std::vector<size_t> get_soln_sizes(PDE<prec> const &pde)
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

std::vector<std::vector<prec>> nodes_;
auto sol_sizes_ = std::vector<size_t>(1);
auto elem_coords_ = std::vector<std::vector<prec>>(1);

void init_plotting(PDE<prec> const &pde, elements::table const &table)
{
  // Generates cell array of nodes and element coordinates needed for plotting
  sol_sizes_ = get_soln_sizes(pde);

  nodes_ = std::vector<std::vector<prec>>(pde.num_dims);
  auto const dims = pde.get_dimensions();

  for (int i = 0; i < pde.num_dims; i++)
  {
    auto const &dim = dims[i];
    auto const &node_list = generate_nodes(dim.get_degree(), dim.get_level(),
                                           dim.domain_min, dim.domain_max);
    nodes_[i] = node_list;
  }

  elem_coords_ = gen_elem_coords(pde, table);

  std::cout << "Attempted to plot with " << elem_coords_.size() << ", " << elem_coords_[0].size() << "\n";
}

void plot_fval(PDE<prec> const &pde, elements::table const &table,
               fk::vector<prec> const &f_val,
               fk::vector<prec> const &analytic_soln)
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

INJECTION_TEST(MeshWatcher, AdaptivityTracking)
{
  // Can't use prec for type parameter in these two GetRef conversions.
  auto &grid = GetRef("adaptive_grid", adapt::distributed_grid<double>);
  auto &pde  = GetRef("pde_deref", PDE<double>*);
  engine->Put("Elements", grid.size());
  engine->Put("Labels", stageId);
  if (stageId == "Plot Data" && pde->has_analytic_soln) {
    auto &real_space = GetRef("real_space", fk::vector<prec>);
    auto &analytic_solution_real_space = GetRef("analytic_solution_realspace", fk::vector<prec>);
    plot_fval(*pde, grid.get_table(), real_space, analytic_solution_real_space);
  }
  return SUCCESS;
}

#endif
