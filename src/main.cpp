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
INJECTION_EXECUTABLE(ASGARD)
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

  //Initialize VnV
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
  node_out() << "generating: pde..." << '\n';
  auto pde = make_PDE<prec>(cli_input);

  // do this only once to avoid confusion
  // if we ever do go to p-adaptivity (variable degree) we can change it then
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

  // -- create forward/reverse mapping between elements and indices,
  // -- along with a distribution plan. this is the adaptive grid.
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
  VnV_Info(ASGARD, "Generating Initial Conditions");
  auto const initial_condition =
      adaptive_grid.get_initial_condition(*pde, transformer, opts);
  VnV_Info(ASGARD, "Degrees of freedom (post initial adapt): %ld ",
           adaptive_grid.size() *
               static_cast<uint64_t>(std::pow(degree, pde->num_dims)));

  // -- generate and store coefficient matrices.
  VnV_Info(ASGARD, "Generating Coeffcient Matrices");
  generate_all_coefficients<prec>(*pde, transformer);

  // this is to bail out for further profiling/development on the setup routines
  if (opts.num_time_steps < 1)
    return 0;

  node_out() << "--- begin time loop staging ---" << '\n';

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

// -- setup realspace transform for file io or for plotting
#if defined(ASGARD_IO_HIGHFIVE) || defined(ASGARD_USE_MATLAB) || defined(ASGARD_USE_VNV)

  // realspace solution vector - WARNING this is
  // currently infeasible to form for large problems
  auto const real_space_size = real_solution_size(*pde);
  fk::vector<prec> real_space(real_space_size);

  // temporary workspaces for the transform
  fk::vector<prec, mem_type::owner, resource::host> workspace(real_space_size *
                                                              2);
  std::array<fk::vector<prec, mem_type::view, resource::host>, 2>
      tmp_workspace = {
          fk::vector<prec, mem_type::view, resource::host>(workspace, 0,
                                                           real_space_size),
          fk::vector<prec, mem_type::view, resource::host>(
              workspace, real_space_size, real_space_size * 2 - 1)};
  // transform initial condition to realspace
  wavelet_to_realspace<prec>(*pde, initial_condition, adaptive_grid.get_table(),
                             transformer, default_workspace_cpu_MB,
                             tmp_workspace, real_space);
#endif

#ifdef ASGARD_USE_MATLAB
  ml::matlab_plot ml_plot;
  ml_plot.connect(cli_input.get_ml_session_string());
  node_out() << "  connected to MATLAB" << '\n';
#endif

#if defined(ASGARD_USE_MATLAB) || defined(ASGARD_USE_VNV)
  fk::vector<prec> analytic_solution_realspace(real_space_size);
  if (pde->has_analytic_soln)
  {
    // generate the analytic solution at t=0
    auto const subgrid_init           = adaptive_grid.get_subgrid(get_rank());
    auto const analytic_solution_init = transform_and_combine_dimensions(
        *pde, pde->exact_vector_funcs, adaptive_grid.get_table(), transformer,
        subgrid_init.col_start, subgrid_init.col_stop, degree);
    // transform analytic solution to realspace
    wavelet_to_realspace<prec>(
        *pde, analytic_solution_init, adaptive_grid.get_table(), transformer,
        default_workspace_cpu_MB, tmp_workspace, analytic_solution_realspace);
  }
#endif

#ifdef ASGARD_USE_MATLAB
  // Add the matlab scripts directory to the matlab path
  ml_plot.add_param(std::string(ASGARD_SCRIPTS_DIR) + "matlab/");
  ml_plot.call("addpath");

  ml_plot.init_plotting(*pde, adaptive_grid.get_table());
  ml_plot.plot_fval(*pde, adaptive_grid.get_table(), real_space,
                    analytic_solution_realspace);
#endif

  // -- setup output file and write initial condition
#ifdef ASGARD_IO_HIGHFIVE
  // initialize wavelet output
  auto output_dataset = initialize_output_file(initial_condition);

  // initialize realspace output
  auto const realspace_output_name = "asgard_realspace";
  auto output_dataset_real =
      initialize_output_file(real_space, "asgard_realspace");
#endif

  // -- time loop

  fk::vector<prec> f_val(initial_condition);
  node_out() << "--- begin time loop w/ dt " << pde->get_dt() << " ---\n";

  node_out() << "adaptive grid " << adaptive_grid.size() << " " << &adaptive_grid << "\n";

  auto const method = opts.use_implicit_stepping
                          ? time_advance::method::imp
                          : time_advance::method::exp;
  int i;
  double time;
  int sol_size;

  /**
   * Asgard Time-Stepping Loop with Mesh Adaptivity
   * ==============================================
   *
   * This is the main time-stepping loop in the ASGarD executable. The loop
   * will execute :vnv:`nts[0]` time steps.
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
          engine->Put("time", time);

          // print root mean squared error from analytic solution
          // @BEN Make this a VnV Test
          if (pde->has_analytic_soln)
          {
            // get analytic solution at time(step+1)
            auto const subgrid= adaptive_grid.get_subgrid(get_rank());
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

            VnV_Info(ASGARD, "Transforming to real space (analytic)");

            auto transform_wksp = update_transform_workspace<prec>(
                sol_size, workspace, tmp_workspace);
            if (analytic_solution.size() > analytic_solution_realspace.size())
            {
              analytic_solution_realspace.resize(analytic_solution.size());
            }
            wavelet_to_realspace<prec>(
                *pde, analytic_solution, adaptive_grid.get_table(),
                transformer, default_workspace_cpu_MB, transform_wksp,
                analytic_solution_realspace);

            VnV_Info(ASGARD, "Transformed to real space (analytic)");
          }
          else
          {
            node_out() << "No analytic solution found." << '\n';
          }

          VnV_Info(ASGARD, "Transforming to real space");

          /* transform from wavelet space to real space */
          // resize transform workspaces if grid size changed due to adaptivity
          auto const real_size = real_solution_size(*pde);
          auto transform_wksp  = update_transform_workspace<prec>(
              real_size, workspace, tmp_workspace);
          real_space.resize(real_size);

          wavelet_to_realspace<prec>(*pde, f_val, adaptive_grid.get_table(),
                                     transformer, default_workspace_cpu_MB,
                                     transform_wksp, real_space);

          VnV_Info(ASGARD, "Transformed to real space");

#ifdef ASGARD_IO_HIGHFIVE
          // write output to file
          if (opts.should_output_wavelet(i))
          {
            update_output_file(output_dataset, f_val);
          }
          if (opts.should_output_realspace(i))
          {
            update_output_file(output_dataset_real, real_space,
                               realspace_output_name);
          }
#else
          ignore(default_workspace_cpu_MB);
#endif

#ifdef ASGARD_USE_MATLAB
          if (opts.should_plot(i))
          {
            ml_plot.plot_fval(*pde, adaptive_grid.get_table(), real_space,
                              analytic_solution_realspace);
          }
#endif
        }},
      adaptive_grid, pde, analytic_solution_realspace, real_space);

  for (i = 0; i < opts.num_time_steps; ++i)
  {
    // take a time advance step
    time = (i + 1) * pde->get_dt();
    auto const update_system = i == 0;
    auto const time_str = opts.use_implicit_stepping
                                   ? "implicit_time_advance"
                                   : "explicit_time_advance";

    auto const time_id = tools::timer.start(time_str);
    auto const sol = time_advance::adaptive_advance(
                  method, *pde, adaptive_grid, transformer, opts, f_val, time,
                  default_workspace_MB, update_system);

    sol_size = sol.size();
    f_val.resize(sol_size) = sol;
    tools::timer.stop(time_id);

    INJECTION_LOOP_ITER("ASGARD", "TimeStepping", "TS " + std::to_string(i));

    node_out() << "timestep: " << i << " complete" << '\n';
  }
  
  INJECTION_LOOP_END("ASGARD", "TimeStepping");

  node_out() << "--- simulation complete ---" << '\n';

  auto const segment_size = element_segment_size(*pde);

  // gather results from all ranks. not currently writing the result anywhere
  // yet, but rank 0 holds the complete result after this call
  auto const final_result = gather_results(
      f_val, adaptive_grid.get_distrib_plan(), my_rank, segment_size);

  node_out() << tools::timer.report() << '\n';

  INJECTION_FINALIZE(ASGARD)
  finalize_distribution();

  return 0;
}
