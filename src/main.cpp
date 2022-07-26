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
 *           "adaptive_threshold" : 0.001,
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
 *               "ASGARD:MeshInfo" : {}
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
  )")
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
  INJECTION_LOOP_BEGIN_C(ASGARD, VASGARD, Configuration, IPCALLBACK {
        if (type == VnV::InjectionPointType::Begin)
        {
          VnV::MetaData d;
          d["table"] = "build";
          engine->Put("Commit Branch", GIT_BRANCH, d);
          engine->Put("Commit Summary", GIT_COMMIT_SUMMARY, d);
          engine->Put("Commit Hash", GIT_COMMIT_HASH, d);
          engine->Put("Build Time", BUILD_TIME, d);

          d["table"] = "run";
          engine->Put("PDE", cli_input.get_pde_string(), d);
          engine->Put("Time Steps: ", opts.num_time_steps, d);
          engine->Put("Write freq: ", opts.wavelet_output_freq, d);
          engine->Put("Realspace freq: ", opts.realspace_output_freq, d);
          engine->Put("Implicit Stepper: ", opts.use_implicit_stepping, d);
          engine->Put("Full grid: ", opts.use_full_grid, d);
          engine->Put("CFL number: ", cli_input.get_cfl(), d);
          engine->Put("Poisson solve: ", opts.do_poisson_solve, d);
          engine->Put("Maximum adaptivity levels: ", opts.max_level, d);
        }
      },
      opts, cli_input);

  INJECTION_LOOP_ITER(ASGARD, Configuration, Generate PDE);
  // -- generate pde
  auto pde = make_PDE<prec>(cli_input);

  // -- set degree (constant since no p-adaptivity)
  auto degree = pde->get_dimensions()[0].get_degree();

  INJECTION_LOOP_ITER(ASGARD, Configuration, Generate Adaptive Grid);
  // -- create forward/reverse mapping between elements and indices,
  // -- along with a distribution plan. this is the adaptive grid.
  adapt::distributed_grid adaptive_grid(*pde, opts);

  INJECTION_LOOP_ITER(ASGARD, Configuration, Generate Basis Operator);
  // -- generate basis operator
  auto const quiet = false;
  basis::wavelet_transform<prec, resource::host> const transformer(opts, *pde, quiet);
 
  INJECTION_LOOP_ITER(ASGARD, Configuration, Generate Mass Matrices);
  // -- generate and store the mass matrices for each dimension
  generate_dimension_mass_mat<prec>(*pde, transformer);
  
  INJECTION_LOOP_ITER(ASGARD, Configuration, Generate IC Vector);
  // -- generate initial condition vector
  auto const initial_condition = adaptive_grid.get_initial_condition(*pde, transformer, opts);
  //degrees of freedom (post initial adapt)
  //adaptive_grid.size() * static_cast<uint64_t>(std::pow(degree, pde->num_dims))

  INJECTION_LOOP_ITER(ASGARD, Configuration, Regenerate Mass Matrices);
  // -- regen mass mats after init conditions - TODO: check dims/rechaining?
  generate_dimension_mass_mat<prec>(*pde, transformer);

  INJECTION_LOOP_ITER(ASGARD, Configuration, Generate Coeff Matrix);
  // -- generate and store coefficient matrices.
  generate_all_coefficients<prec>(*pde, transformer);

  // this is to bail out for further profiling/development on the setup routines
  if (opts.num_time_steps < 1)
    return 0;

  INJECTION_LOOP_END(ASGARD, Configuration);


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

  // -- time loop

  fk::vector<prec> f_val(initial_condition);

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
  INJECTION_LOOP_BEGIN_C(ASGARD, VASGARD, TimeStepping, IPCALLBACK {

    if (type == VnV::InjectionPointType::Begin) {
      engine->Put("timesteps", opts.num_time_steps);
      engine->Put("impicit", opts.use_implicit_stepping ? "implicit" : "explicit");
    }
  }, adaptive_grid, pde, opts, time, f_val, transformer);

  
  for (auto i = 0; i < opts.num_time_steps; ++i)
  {
    // take a time advance step
    time = (i + 1) * pde->get_dt();
    //FIXME provide updated adaptive_grid, pde, f_val, and transformer to avoid error with plotting in --adapt

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
    }
    else
    {
      node_out() << "No analytic solution found." << '\n';
    }

    INJECTION_LOOP_ITER_D(ASGARD, TimeStepping, "TS " + std::to_string(i));
  }
  INJECTION_LOOP_END(ASGARD, TimeStepping);

  auto const segment_size = element_segment_size(*pde);

  // gather results from all ranks. not currently writing the result anywhere
  // yet, but rank 0 holds the complete result after this call
  auto const final_result = gather_results(
      f_val, adaptive_grid.get_distrib_plan(), my_rank, segment_size);

  INJECTION_FINALIZE(ASGARD)

  finalize_distribution();

  return 0;
}

