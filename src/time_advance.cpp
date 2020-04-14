#include "time_advance.hpp"
#include "boundary_conditions.hpp"
#include "distribution.hpp"
#include "element_table.hpp"
#include "fast_math.hpp"
#include "solver.hpp"
#include "timer.hpp"
#include <limits.h>

// this function executes an explicit time step using the current solution
// vector x. on exit, the next solution vector is stored in x.
template<typename P>
fk::vector<P>
explicit_time_advance(PDE<P> const &pde, element_table const &table,
                      std::vector<fk::vector<P>> const &unscaled_sources,
                      std::array<unscaled_bc_parts<P>, 2> const &unscaled_parts,
                      fk::vector<P> const &x,
                      std::vector<element_chunk> const &chunks,
                      distribution_plan const &plan, P const time, P const dt)
{
  int const my_rank   = get_rank();
  auto const &grid    = plan.at(get_rank());
  int const elem_size = element_segment_size(pde);

  // allocate workspace
  int64_t const col_size = elem_size * static_cast<int64_t>(grid.ncols());
  assert(x.size() == col_size);
  int64_t const row_size = elem_size * static_cast<int64_t>(grid.nrows());
  assert(col_size < INT_MAX);
  assert(row_size < INT_MAX);

  // input vector for apply_A
  fk::vector<P> fx(col_size);
  fm::copy(x, fx);

  // a buffer for reducing across subgrid row
  fk::vector<P> reduced_fx(row_size);

  assert(time >= 0);
  assert(dt > 0);
  assert(static_cast<int>(unscaled_sources.size()) == pde.num_sources);

  // see
  // https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods#Explicit_Runge%E2%80%93Kutta_methods
  P const a21 = 0.5;
  P const a31 = -1.0;
  P const a32 = 2.0;
  P const b1  = 1.0 / 6.0;
  P const b2  = 2.0 / 3.0;
  P const b3  = 1.0 / 6.0;
  P const c2  = 1.0 / 2.0;
  P const c3  = 1.0;

  auto const apply_id = timer::record.start("apply_A");
  auto fx1            = apply_A(pde, table, grid, chunks, fx);
  timer::record.stop(apply_id);

  reduce_results(fx1, reduced_fx, plan, my_rank);

  if (!unscaled_sources.empty())
  {
    auto const scaled_source = scale_sources(pde, unscaled_sources, time);
    fm::axpy(scaled_source, reduced_fx);
  }

  auto const bc0 = boundary_conditions::generate_scaled_bc(
      unscaled_parts[0], unscaled_parts[1], pde, grid.row_start, grid.row_stop,
      time);

  fm::axpy(bc0, reduced_fx);

  exchange_results(reduced_fx, fx1, elem_size, plan, my_rank);

  P const fx_scale_1 = a21 * dt;
  fm::axpy(fx1, fx, fx_scale_1);

  timer::record.start(apply_id);
  auto fx2 = apply_A(pde, table, grid, chunks, fx);
  timer::record.stop(apply_id);

  reduce_results(fx2, reduced_fx, plan, my_rank);

  if (!unscaled_sources.empty())
  {
    auto const scaled_source =
        scale_sources(pde, unscaled_sources, time + c2 * dt);
    fm::axpy(scaled_source, reduced_fx);
  }

  fk::vector<P> const bc1 = boundary_conditions::generate_scaled_bc(
      unscaled_parts[0], unscaled_parts[1], pde, grid.row_start, grid.row_stop,
      time + c2 * dt);

  fm::axpy(bc1, reduced_fx);

  exchange_results(reduced_fx, fx2, elem_size, plan, my_rank);

  fm::copy(x, fx);
  P const fx_scale_2a = a31 * dt;
  P const fx_scale_2b = a32 * dt;

  fm::axpy(fx1, fx, fx_scale_2a);
  fm::axpy(fx2, fx, fx_scale_2b);

  timer::record.start(apply_id);
  auto fx3 = apply_A(pde, table, grid, chunks, fx);
  timer::record.stop(apply_id);

  reduce_results(fx3, reduced_fx, plan, my_rank);

  if (!unscaled_sources.empty())
  {
    auto const scaled_source =
        scale_sources(pde, unscaled_sources, time + c3 * dt);
    fm::axpy(scaled_source, reduced_fx);
  }
  auto const bc2 = boundary_conditions::generate_scaled_bc(
      unscaled_parts[0], unscaled_parts[1], pde, grid.row_start, grid.row_stop,
      time + c3 * dt);

  fm::axpy(bc2, reduced_fx);
  exchange_results(reduced_fx, fx3, elem_size, plan, my_rank);

  fm::copy(x, fx);
  P const scale_1 = dt * b1;
  P const scale_2 = dt * b2;
  P const scale_3 = dt * b3;

  fm::axpy(fx1, fx, scale_1);
  fm::axpy(fx2, fx, scale_2);
  fm::axpy(fx3, fx, scale_3);

  return fx;
}

// scale source vectors for time
template<typename P>
fk::vector<P>
scale_sources(PDE<P> const &pde,
              std::vector<fk::vector<P>> const &unscaled_sources, P const time)
{
  // zero out final vect
  assert(unscaled_sources.size() > 0);
  fk::vector<P> scaled_source(unscaled_sources[0].size());

  // scale and accumulate all sources
  for (int i = 0; i < pde.num_sources; ++i)
  {
    fm::axpy(unscaled_sources[i], scaled_source,
             pde.sources[i].time_func(time));
  }
  return scaled_source;
}

// this function executes an implicit time step using the current solution
// vector x. on exit, the next solution vector is stored in fx.
template<typename P>
fk::vector<P>
implicit_time_advance(PDE<P> const &pde, element_table const &table,
                      std::vector<fk::vector<P>> const &unscaled_sources,
                      std::array<unscaled_bc_parts<P>, 2> const &unscaled_parts,
                      fk::vector<P> const &x_orig,
                      std::vector<element_chunk> const &chunks,
                      distribution_plan const &plan, P const time, P const dt,
                      solve_opts const solver, bool const update_system)
{
  assert(time >= 0);
  assert(dt > 0);
  assert(static_cast<int>(unscaled_sources.size()) == pde.num_sources);
  static fk::matrix<P, mem_type::owner, resource::host> A;
  static std::vector<int> ipiv;
  static bool first_time = true;

  int const degree    = pde.get_dimensions()[0].get_degree();
  int const elem_size = static_cast<int>(std::pow(degree, pde.num_dims));
  int const A_size    = elem_size * table.size();

  auto const scaled_source = scale_sources(pde, unscaled_sources, time + dt);

  auto x = x_orig + (scaled_source * dt);

  /* add the boundary condition */
  int const my_rank = get_rank();
  auto const &grid  = plan.at(my_rank);

  auto const bc = boundary_conditions::generate_scaled_bc(
      unscaled_parts[0], unscaled_parts[1], pde, grid.row_start, grid.row_stop,
      time + dt);
  fm::axpy(bc, x, dt);

  if (first_time || update_system)
  {
    A.clear_and_resize(A_size, A_size);
    for (auto const &chunk : chunks)
    {
      build_system_matrix(pde, table, chunk, A);
    }

    // AA = I - dt*A;
    for (int i = 0; i < A.nrows(); ++i)
    {
      for (int j = 0; j < A.ncols(); ++j)
      {
        A(i, j) *= -dt;
      }
      A(i, i) += 1.0;
    }

    switch (solver)
    {
    case solve_opts::direct:
      if (ipiv.size() != static_cast<unsigned long>(A.nrows()))
        ipiv.resize(A.nrows());
      fm::gesv(A, x, ipiv);
      return x;
      break;
    case solve_opts::gmres:
      ignore(ipiv);
      break;
    }

    first_time = false;
  } // end first time/update system

  switch (solver)
  {
  case solve_opts::direct:
    fm::getrs(A, x, ipiv);
    return x;
    break;
  case solve_opts::gmres:
    P const tolerance  = std::is_same_v<float, P> ? 1e-6 : 1e-12;
    int const restart  = A.ncols();
    int const max_iter = A.ncols();
    fk::vector<P> fx(x.size());
    solver::simple_gmres(A, fx, x, fk::matrix<P>(), restart, max_iter,
                         tolerance);
    return fx;
    break;
  }
  return x;
}

template fk::vector<double> explicit_time_advance(
    PDE<double> const &pde, element_table const &table,
    std::vector<fk::vector<double>> const &unscaled_sources,
    std::array<unscaled_bc_parts<double>, 2> const &unscaled_parts,
    fk::vector<double> const &x, std::vector<element_chunk> const &chunks,
    distribution_plan const &plan, double const time, double const dt);

template fk::vector<float> explicit_time_advance(
    PDE<float> const &pde, element_table const &table,
    std::vector<fk::vector<float>> const &unscaled_sources,
    std::array<unscaled_bc_parts<float>, 2> const &unscaled_parts,
    fk::vector<float> const &x, std::vector<element_chunk> const &chunks,
    distribution_plan const &plan, float const time, float const dt);

template fk::vector<double> implicit_time_advance(
    PDE<double> const &pde, element_table const &table,
    std::vector<fk::vector<double>> const &unscaled_sources,
    std::array<unscaled_bc_parts<double>, 2> const &unscaled_parts,
    fk::vector<double> const &host_space,
    std::vector<element_chunk> const &chunks, distribution_plan const &plan,
    double const time, double const dt, solve_opts const solver,
    bool const update_system);

template fk::vector<float> implicit_time_advance(
    PDE<float> const &pde, element_table const &table,
    std::vector<fk::vector<float>> const &unscaled_sources,
    std::array<unscaled_bc_parts<float>, 2> const &unscaled_parts,
    fk::vector<float> const &x, std::vector<element_chunk> const &chunks,
    distribution_plan const &plan, float const time, float const dt,
    solve_opts const solver, bool const update_system);
