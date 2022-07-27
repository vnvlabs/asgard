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

