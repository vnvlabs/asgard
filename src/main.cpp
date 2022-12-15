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

/**
 * @title Adaptive Sparse Grid Discretization with ASGARD
 * @description The Main Asgard Application
 * @shortTitle ASGARD
 * @configuration { 
 * "input" : {
 *     "outputEngine": {
 *       "json_file": {
 *           "filename": "aout"
 *       }
 *   },
 *   "options":{
 *       "ASGARD" : {
 *           "adapt" : true,
 *           "adaptive_threshold" : 0.1,
 *           "max_levels" : 8
 *       }
 *   },
 *   "injectionPoints" : {
 *       "ASGARD:Configuration" : {
 *           "tests" : {
 *               "VNV:cputime" : {}
 *           }
 *       },
 *       "ASGARD:TimeStepping" : {
 *           "tests" : {
 *               "ASGARD:PlotSolution" : {},
 *               "ASGARD:MeshInfo" : {},
 *               "ASGARD:PlotError" : {}
 *           }
 *       }
 *    }
 *  }
 * }
 * 
 * 
 * Adaptive Sparse Grid Discretization:
 * ====================================
 * 
 * Many scientific domains require the solution of
 * high dimensional PDEs. Traditional grid- or mesh-based methods for solving
 * such systems in a noise-free manner quickly become intractable due to the
 * scaling of the degrees of freedom going as O(N^d) sometimes called "the curse
 * of dimensionality." This application implements  an arbitrarily high-order
 * discontinuous-Galerkin finite-element solver that leverages an adaptive
 * sparse-grid discretization whose degrees of freedom scale as O(N*log2 N^D-1).
 * This method and its subsequent reduction in the required resources is being
 * applied to several PDEs including time-domain Maxwell's equations (3D), the
 * Vlasov equation (in up to 6D) and a Fokker-Planck-like problem in ongoing
 * related efforts.
 *
 * This implementation is designed to run on multiple accelerated architectures,
 * including distributed systems. The implementation takes advantage of a system
 * matrix decomposed as the Kronecker product of many smaller matrices which is
 * implemented as batched operations.
 *
 */
INJECTION_EXECUTABLE(ASGARD)
INJECTION_SUBPACKAGE(ASGARD, ASGARD_time_advance)
INJECTION_SUBPACKAGE(ASGARD, ASGARD_pde)
INJECTION_SUBPACKAGE(ASGARD, ASGARD_tools)

class AsgardOptions
{
  nlohmann::json jj;

public:
  AsgardOptions(const nlohmann::json &j) : jj(j) {}
};

/**
 * @title Asgard VnV Options:
 *
 * Asgard Options Information
 * --------------------------
 *
 * TODO Write out some information about ASGARD HERE -- Not really sure what goes here to be 
 * honest. 
 * 
 */
INJECTION_OPTIONS(ASGARD, R"(
  {
   "type" : "object",
   "properties" : {
       "cfl" : { "type" : "number" , "default" : 0.01, "min" : 1e-10, "max":5,  "description" : "What CFL number should we target?"},
       "adaptive_threshold" : { "type" : "number" , "default" : 0.25 , "min" : 0,  "description" : "The threshold for adaption" },
       "max_levels" : {"type" : "integer" , "default" : 8, "min" : 0, "description" : "Maximum number of levels" },
       "time_steps" : {"type" : "integer" , "default" : 6, "min" : 1 ,  "description" : "How many time steps should we take?"},
       "adapt" : {"type" : "boolean" , "default" : true,  "description" : "Use Adaptive Grids?"},
       "poisson" : {"type" : "boolean" , "default" : false,  "description" : "Use Poisson?"},
       "implicit" : {"type" : "boolean" , "default" : false, "description" : "Use an implicit time stepping algorithm?" }}
  }
  )", AsgardOptions)
{
  return new AsgardOptions(config);
}

int main(int argc, char **argv)
{
  // -- set up distribution
  auto const [my_rank, num_ranks] = initialize_distribution();

  // -- parse cli
  parser const cli_input(argc, argv);
  if (!cli_input.is_valid())
  {
    exit(-1);
  }

  /**
   * @title Asgard Application. 
   * @shortTitle Asgard Main.
   * 
   * Many scientific domains require the solution of
   * high dimensional PDEs. Traditional grid- or mesh-based methods for solving
   * such systems in a noise-free manner quickly become intractable due to the
   * scaling of the degrees of freedom going as O(N^d) sometimes called "the curse
   * of dimensionality." This application implements  an arbitrarily high-order
   * discontinuous-Galerkin finite-element solver that leverages an adaptive
   * sparse-grid discretization whose degrees of freedom scale as O(N*log2 N^D-1).
   * This method and its subsequent reduction in the required resources is being
   * applied to several PDEs including time-domain Maxwell's equations (3D), the
   * Vlasov equation (in up to 6D) and a Fokker-Planck-like problem in ongoing
   * related efforts.
   *
   * This implementation is designed to run on multiple accelerated architectures,
   * including distributed systems. The implementation takes advantage of a system
   * matrix decomposed as the Kronecker product of many smaller matrices which is
   * implemented as batched operations.
   * 
   */
  INJECTION_INITIALIZE(ASGARD, &argc, &argv, "./vv-input.json");

  options const opts(cli_input);

  // kill off unused processes
  if (my_rank >= num_ranks)
  {
    INJECTION_FINALIZE(ASGARD)
    finalize_distribution();
    return 0;
  }

  /**
   * @title Application Configuration:
   * @shortTitle Configuration
   * @description Configuration stage of the Asgard process.
   * 
   * 
   * Compilation information
   * -----------------------
   *
   * .. vnv-quick-table::
   *    :names: ["Property", "Value"]
   *    :fields: ["name", "value"]
   *    :data: *|[?_table==`build`].{ "name" : Name , "value" : Value }
   *
   *
   * Application Configuration
   * --------------------------
   *
   * .. vnv-quick-table::
   *    :names: ["Property", "Value"]
   *    :fields: ["name", "value"]
   *    :data: *|[?_table==`run`].{ "name" : Name , "value" : Value }
   *
   * 
   */
  INJECTION_LOOP_BEGIN(ASGARD, VASGARD, Configuration, VNV_CALLBACK {
 
          VnV::MetaData d;
          d["table"] = "build";
         data.engine->Put("Commit Branch", GIT_BRANCH, d);
         data.engine->Put("Commit Summary", GIT_COMMIT_SUMMARY, d);
         data.engine->Put("Commit Hash", GIT_COMMIT_HASH, d);
         data.engine->Put("Build Time", BUILD_TIME, d);

          d["table"] = "run";
         data.engine->Put("PDE", cli_input.get_pde_string(), d);
         data.engine->Put("Time Steps: ", opts.num_time_steps, d);
         data.engine->Put("Write freq: ", opts.wavelet_output_freq, d);
         data.engine->Put("Realspace freq: ", opts.realspace_output_freq, d);
         data.engine->Put("Implicit Stepper: ", opts.use_implicit_stepping, d);
         data.engine->Put("Full grid: ", opts.use_full_grid, d);
         data.engine->Put("CFL number: ", cli_input.get_cfl(), d);
         data.engine->Put("Poisson solve: ", opts.do_poisson_solve, d);
         data.engine->Put("Maximum adaptivity levels: ", opts.max_level, d);
        
      },
      opts, cli_input);

  // -- generate pde
  INJECTION_LOOP_ITER(ASGARD, Configuration, "Generate PDE",VNV_NOCALLBACK);
  auto pde = make_PDE<prec>(cli_input);

  // -- set degree (constant since no p-adaptivity)
  auto degree = pde->get_dimensions()[0].get_degree();

  // -- create forward/reverse mapping between elements and indices,
  // -- along with a distribution plan. this is the adaptive grid.
  INJECTION_LOOP_ITER(ASGARD, Configuration, "Generate Adaptive Grid",VNV_NOCALLBACK);
  adapt::distributed_grid adaptive_grid(*pde, opts);

  // -- generate basis operator
  INJECTION_LOOP_ITER(ASGARD, Configuration, "Generate Basis Operator",VNV_NOCALLBACK);
  auto const quiet = false;
  basis::wavelet_transform<prec, resource::host> const transformer(opts, *pde, quiet);
 
  // -- generate and store the mass matrices for each dimension
  INJECTION_LOOP_ITER(ASGARD, Configuration, "Generate Mass Matrices",VNV_NOCALLBACK);
  generate_dimension_mass_mat<prec>(*pde, transformer);
  
  // -- generate initial condition vector
  INJECTION_LOOP_ITER(ASGARD, Configuration, "Generate IC Vector",VNV_NOCALLBACK);
  auto const initial_condition = adaptive_grid.get_initial_condition(*pde, transformer, opts);
  //degrees of freedom (post initial adapt)
  //adaptive_grid.size() * static_cast<uint64_t>(std::pow(degree, pde->num_dims))

  // -- regen mass mats after init conditions - TODO: check dims/rechaining?
  INJECTION_LOOP_ITER(ASGARD, Configuration, "Regenerate Mass Matrices",VNV_NOCALLBACK);
  generate_dimension_mass_mat<prec>(*pde, transformer);

  // -- generate and store coefficient matrices.
  INJECTION_LOOP_ITER(ASGARD, Configuration, "Generate Coeff Matrix",VNV_NOCALLBACK);
  generate_all_coefficients<prec>(*pde, transformer);

  // this is to bail out for further profiling/development on the setup routines
  if (opts.num_time_steps < 1)
    return 0;

  INJECTION_LOOP_END(ASGARD, Configuration,VNV_NOCALLBACK);

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
#if defined(ASGARD_IO_HIGHFIVE) || defined(ASGARD_USE_MATLAB)

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
                                                           real_space_size - 1),
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

  // Add the matlab scripts directory to the matlab path
  ml_plot.add_param(std::string(ASGARD_SCRIPTS_DIR) + "matlab/");
  ml_plot.call("addpath");

  ml_plot.init_plotting(*pde, adaptive_grid.get_table());
  ml_plot.plot_fval(*pde, adaptive_grid.get_table(), real_space,
                    analytic_solution_realspace);

  // send initial condition to matlab
  std::vector<size_t> sizes(pde->num_dims);
  for (int i = 0; i < pde->num_dims; i++)
  {
    sizes[i] = pde->get_dimensions()[i].get_degree() *
               fm::two_raised_to(pde->get_dimensions()[i].get_level());
  }
  ml_plot.set_var("initial_condition",
                  ml_plot.create_array(sizes, initial_condition));

  ml_plot.copy_pde(*pde);
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

  prec time = 0;

  /**
   * @title ASGARD Time Stepping Loop
   * @shortTitle Step
   * @description The Core time stepping loop
   *
   * In this section of the code we propergate the initial condition forward in
   * time using :vnv:`timesteps` timesteps of an :vnv:`implicit` time stepping 
   * routine.  
   * 
   */
  INJECTION_LOOP_BEGIN(ASGARD, VASGARD, TimeStepping, VNV_CALLBACK {

     data.engine->Put("timesteps", opts.num_time_steps);
     data.engine->Put("impicit", opts.use_implicit_stepping ? "implicit" : "explicit");
  
  }, adaptive_grid, pde, opts, time, f_val, transformer);
  
  for (auto i = 0; i < opts.num_time_steps; ++i)
  {
    // take a time advance step
    time = (i + 1) * pde->get_dt();

    auto const update_system = i == 0;
    auto const method   = opts.use_implicit_stepping ? time_advance::method::imp
                                                     : time_advance::method::exp;
    auto const time_str = opts.use_implicit_stepping ? "implicit_time_advance"
                                                     : "explicit_time_advance";
    auto const time_id  = tools::timer.start(time_str);
    auto const sol      = time_advance::adaptive_advance(
        method, *pde, adaptive_grid, transformer, opts, f_val, time,
        default_workspace_MB, update_system);
    f_val.resize(sol.size()) = sol;
    tools::timer.stop(time_id);

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
      auto const relative_error = RMSE / inf_norm(analytic_solution) * 100;
      auto const [rmse_errors, relative_errors] =
          gather_errors(RMSE, relative_error);
      expect(rmse_errors.size() == relative_errors.size());
      for (int i = 0; i < rmse_errors.size(); ++i)
      {
        node_out() << "Errors for local rank: " << i << '\n';
        node_out() << "RMSE (numeric-analytic) [wavelet]: " << rmse_errors(i)
                   << '\n';
        node_out() << "Relative difference (numeric-analytic) [wavelet]: "
                   << relative_errors(i) << " %" << '\n';
      }

#ifdef ASGARD_USE_MATLAB
      if (opts.should_plot(i))
      {
        auto const real_size = real_solution_size(*pde);
        auto transform_wksp  = update_transform_workspace<prec>(
            real_size, workspace, tmp_workspace);
        if (real_size > analytic_solution_realspace.size())
        {
          analytic_solution_realspace.resize(real_size);
        }
        wavelet_to_realspace<prec>(*pde, analytic_solution,
                                   adaptive_grid.get_table(), transformer,
                                   default_workspace_cpu_MB, transform_wksp,
                                   analytic_solution_realspace);
      }
#endif
    }
    else
    {
      node_out() << "No analytic solution found." << '\n';
    }
#if defined(ASGARD_IO_HIGHFIVE) || defined(ASGARD_USE_MATLAB)
    /* transform from wavelet space to real space */
    if (opts.should_output_realspace(i) || opts.should_plot(i))
    {
      // resize transform workspaces if grid size changed due to adaptivity
      auto const real_size = real_solution_size(*pde);
      auto transform_wksp =
          update_transform_workspace<prec>(real_size, workspace, tmp_workspace);
      real_space.resize(real_size);

      wavelet_to_realspace<prec>(*pde, f_val, adaptive_grid.get_table(),
                                 transformer, default_workspace_cpu_MB,
                                 transform_wksp, real_space);
    }
#endif

    // write output to file
#ifdef ASGARD_IO_HIGHFIVE
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
    INJECTION_LOOP_ITER(ASGARD, TimeStepping, "TS " + std::to_string(i), VNV_NOCALLBACK);
    node_out() << "timestep: " << i << " complete" << '\n';
  }
  INJECTION_LOOP_END(ASGARD, TimeStepping,VNV_NOCALLBACK);
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
