#include "batch.hpp"

#include "build_info.hpp"
#include "coefficients.hpp"
#include "distribution.hpp"
#include "elements.hpp"
#include "tools.hpp"

#ifdef ASGARD_IO_HIGHFIVE
#include "io.hpp"
#endif

#ifdef ASGARD_USE_MPI
#include <mpi.h>
#endif

#ifdef ASGARD_USE_MATLAB
#include "matlab_plot.hpp"
#endif

#include "pde.hpp"
#include "program_options.hpp"
#include "tensors.hpp"
#include "time_advance.hpp"
#include "transformations.hpp"
#include <numeric>

#ifdef ASGARD_USE_DOUBLE_PREC
using prec = double;
#else
using prec = float;
#endif

#include "asgard_vnv.h"

INJECTION_EXECUTABLE(ASGARD);
INJECTION_SUBPACKAGE(ASGARD, ASGARD_time_advance)
INJECTION_SUBPACKAGE(ASGARD, ASGARD_pde)
INJECTION_SUBPACKAGE(ASGARD, ASGARD_tools)

int main(int argc, char **argv)
{
  // -- set up distribution
  auto const [my_rank, num_ranks] = initialize_distribution();

  // -- parse cli
  parser const cli_input(argc, argv);
  if (!cli_input.is_valid())
  {
    node_out() << "invalid cli string; exiting" << '\n';
    exit(-1);
  }

  // Initialize VnV
  INJECTION_INITIALIZE(ASGARD, &argc, &argv, "./vv-input.json");

  options const opts(cli_input);

  // kill off unused processes
  if (my_rank >= num_ranks)
  {
    INJECTION_FINALIZE(ASGARD)
    finalize_distribution();
    return 0;
  }

  // -- generate pde
  VnV_Info(ASGARD, "Generating PDE...");
  auto pde          = make_PDE<prec>(cli_input);
  auto const degree = pde->get_dimensions()[0].get_degree();

  /**
   * ASGARD Configuration Options
   * ----------------------------
   *
   * .. vnv-options-table::
   *    :all: true
   *
   */
  INJECTION_POINT_C(
      "ASGARD", VWORLD, "RUN",
      IPCALLBACK {
        engine->Put("Branch", GIT_BRANCH);
        engine->Put("Commit", GIT_COMMIT_HASH);
        engine->Put("Summary", GIT_COMMIT_SUMMARY);
        engine->Put("Build", BUILD_TIME);
        engine->Put("PDE", cli_input.get_pde_string());
        engine->Put("degree", degree);
        engine->Put("steps", opts.num_time_steps);
        engine->Put("wfreq", opts.wavelet_output_freq);
        engine->Put("rfreq", opts.realspace_output_freq);
        engine->Put("implicit", opts.use_implicit_stepping);
        engine->Put("full", opts.use_full_grid);
        engine->Put("cfl", cli_input.get_cfl());
        engine->Put("poisson", opts.do_poisson_solve);
        engine->Put("maxLevels", opts.max_level);

        for (auto &it : pde->get_dimensions())
        {
          engine->Put("levels", it.get_level());
        }
      },
      opts);

  VnV_Info(ASGARD, "--- begin setup ---");
  VnV_Info(ASGARD, "Generating Adaptive Grid");

  adapt::distributed_grid adaptive_grid(*pde, opts);

  VnV_Info(ASGARD, "Adaptive Grid has %ld degrees of freedom",
           adaptive_grid.size() *
               static_cast<uint64_t>(std::pow(degree, pde->num_dims)));

  VnV_Info(ASGARD, "Generating Basis Operator");

  auto const quiet = false;
  basis::wavelet_transform<prec, resource::host> const transformer(opts, *pde,
                                                                   quiet);

  // -- generate initial condition vector

  VnV_Info(ASGARD, "Generating Initial Condition");
  auto const initial_condition =
      adaptive_grid.get_initial_condition(*pde, transformer, opts);
  VnV_Info(ASGARD, "Degrees of freedom (post initial adapt): %ld ",
           adaptive_grid.size() *
               static_cast<uint64_t>(std::pow(degree, pde->num_dims)));

  VnV_Info(ASGARD, "Generating Coeffcient Matrices");
  generate_all_coefficients<prec>(*pde, transformer);

  // this is to bail out for further profiling/development on the setup routines
  if (opts.num_time_steps < 1)
    return 0;

  // Our default device workspace size is 10GB - 12 GB DRAM on TitanV
  // - a couple GB for allocations not currently covered by the
  // workspace limit (including working batch).

  // This limit is only for the device workspace - the portion
  // of our allocation that will be resident on an accelerator
  // if the code is built for that.
  //
  // FIXME eventually going to be settable from the cmake
  static auto const default_workspace_MB = 10000;

  // FIXME currently used to check realspace transform only
  /* RAM on fusiont5 */
  static auto const default_workspace_cpu_MB = 187000;

  fk::vector<prec> f_val(initial_condition);

  auto i = 0;

  /**
   * 
   * Asgard Time-Stepping Loop with Mesh Adaptivity
   * ==============================================
   * 
   * This is the main time stepping loop in the asgard executable. The loop
   * will execute :vnv:`nts[0]` time steps
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
   *
   * .. vnv-if:: analytic
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
   * 
   **/
  INJECTION_LOOP_BEGIN_C(
      "ASGARD", VASGARD, "TimeStepping",
      IPCALLBACK {
        engine->Put("elements", adaptive_grid.size());
        if (type == VnV::InjectionPointType::Begin)
        {
          engine->Put("nts", opts.num_time_steps);
          engine->Put("analytic", pde->has_analytic_soln);
        }
        else if (type == VnV::InjectionPointType::Iter)
        {
          const auto time = (i + 1) * pde->get_dt();
          engine->Put("time", time);

          if (pde->has_analytic_soln)
          {
            // get analytic solution at time(step+1)
            auto const subgrid         = adaptive_grid.get_subgrid(get_rank());
            auto const time_multiplier = pde->exact_time(time + pde->get_dt());
            auto const analytic_solution = transform_and_combine_dimensions(
                *pde, pde->exact_vector_funcs, adaptive_grid.get_table(),
                transformer, subgrid.col_start, subgrid.col_stop, degree, time,
                time_multiplier);

            // calculate root mean squared error
            auto const diff = f_val - analytic_solution;
            auto const RMSE = [&diff]() {
              fk::vector<prec> squared(diff);
              std::transform(squared.begin(), squared.end(), squared.begin(),
                             [](prec const &elem) { return elem * elem; });
              auto const mean =
                  std::accumulate(squared.begin(), squared.end(), 0.0) /
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
      },
      adaptive_grid, pde);

  for (; i < opts.num_time_steps; ++i)
  {
    // take a time advance step
    auto const time          = (i + 1) * pde->get_dt();
    auto const update_system = i == 0;
    auto const method = opts.use_implicit_stepping ? time_advance::method::imp
                                                   : time_advance::method::exp;

    auto const sol = time_advance::adaptive_advance(
        method, *pde, adaptive_grid, transformer, opts, f_val, time,
        default_workspace_MB, update_system);
    
    f_val.resize(sol.size()) = sol;

    INJECTION_LOOP_ITER("ASGARD", "TimeStepping", "TS " + std::to_string(i));
  }

  INJECTION_LOOP_END("ASGARD", "TimeStepping");

  INJECTION_FINALIZE(ASGARD)
  finalize_distribution();

  return 0;
}
