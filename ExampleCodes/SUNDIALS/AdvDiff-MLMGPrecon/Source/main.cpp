#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_TimeIntegrator.H>
#include <AMReX_MLABecLaplacian.H>
#include <AMReX_MLMG.H>
#include <AMReX_MultiFabUtil.H>
#ifdef AMREX_USE_HYPRE
#include <AMReX_Hypre.H>
#endif

#include "myfunc.H"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace amrex;

namespace {

struct RlAction {
    int step = 0;
    Real nlscoef = Real(-1.0);
    int max_nonlinear_iters = 0;
    Real eps_lin = Real(-1.0);
    int lsetup_frequency = 0;
};

struct MlmgTelemetry {
    long solve_calls = 0;
    long iterations = 0;
    Real wall_seconds = Real(0.0);
    int last_iterations = 0;
    Real last_residual = Real(-1.0);
};

struct PhiSummary {
    Real min = Real(0.0);
    Real max = Real(0.0);
    Real sum = Real(0.0);
    Real mean = Real(0.0);
    Real l1 = Real(0.0);
    Real l2 = Real(0.0);
    Real linf = Real(0.0);
    bool finite = true;
};

std::string Trim (std::string value)
{
    auto first = std::find_if_not(value.begin(), value.end(),
                                  [](unsigned char c) { return std::isspace(c); });
    auto last = std::find_if_not(value.rbegin(), value.rend(),
                                 [](unsigned char c) { return std::isspace(c); }).base();
    if (first >= last) {
        return {};
    }
    return std::string(first, last);
}

std::string JsonEscape (const std::string& value)
{
    std::ostringstream os;
    for (char c : value) {
        switch (c) {
        case '\\': os << "\\\\"; break;
        case '"': os << "\\\""; break;
        case '\n': os << "\\n"; break;
        case '\r': os << "\\r"; break;
        case '\t': os << "\\t"; break;
        default: os << c; break;
        }
    }
    return os.str();
}

std::vector<RlAction> LoadRlSchedule (const std::string& path)
{
    std::vector<RlAction> schedule;
    if (path.empty()) {
        return schedule;
    }

    std::ifstream input(path);
    if (!input) {
        amrex::Abort("Could not open rl_control.schedule: " + path);
    }

    std::string line;
    int line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        line = Trim(line);
        if (line.empty() || line[0] == '#') {
            continue;
        }

        std::vector<std::string> fields;
        std::stringstream ss(line);
        std::string field;
        while (std::getline(ss, field, ',')) {
            fields.push_back(Trim(field));
        }

        if (fields.size() != 3 && fields.size() != 5) {
            amrex::Abort("Bad rl_control.schedule line " + std::to_string(line_number) +
                         ": expected step,nlscoef,max_nonlinear_iters"
                         "[,eps_lin,lsetup_frequency]");
        }

        if (fields[0].empty() ||
            (!std::isdigit(static_cast<unsigned char>(fields[0][0])) && fields[0][0] != '-'))
        {
            continue;
        }

        RlAction action;
        action.step = std::stoi(fields[0]);
        action.nlscoef = static_cast<Real>(std::stod(fields[1]));
        action.max_nonlinear_iters = std::stoi(fields[2]);
        if (fields.size() == 5) {
            action.eps_lin = static_cast<Real>(std::stod(fields[3]));
            action.lsetup_frequency = std::stoi(fields[4]);
        }
        schedule.push_back(action);
    }

    std::sort(schedule.begin(), schedule.end(),
              [](const RlAction& a, const RlAction& b) { return a.step < b.step; });
    return schedule;
}

SundialsIntegratorStats operator- (const SundialsIntegratorStats& after,
                                   const SundialsIntegratorStats& before)
{
    SundialsIntegratorStats delta;
    delta.valid = after.valid && before.valid;
    delta.last_flag = after.last_flag;
    delta.num_steps = after.num_steps - before.num_steps;
    delta.step_attempts = after.step_attempts - before.step_attempts;
    delta.rhs_evals_0 = after.rhs_evals_0 - before.rhs_evals_0;
    delta.rhs_evals_1 = after.rhs_evals_1 - before.rhs_evals_1;
    delta.num_lin_solv_setups = after.num_lin_solv_setups - before.num_lin_solv_setups;
    delta.num_nonlin_solv_iters =
        after.num_nonlin_solv_iters - before.num_nonlin_solv_iters;
    delta.num_nonlin_solv_conv_fails =
        after.num_nonlin_solv_conv_fails - before.num_nonlin_solv_conv_fails;
    delta.num_step_solve_fails =
        after.num_step_solve_fails - before.num_step_solve_fails;
    delta.num_prec_evals = after.num_prec_evals - before.num_prec_evals;
    delta.num_prec_solves = after.num_prec_solves - before.num_prec_solves;
    delta.num_lin_iters = after.num_lin_iters - before.num_lin_iters;
    delta.num_lin_conv_fails = after.num_lin_conv_fails - before.num_lin_conv_fails;
    delta.num_lin_rhs_evals = after.num_lin_rhs_evals - before.num_lin_rhs_evals;
    delta.hinused = after.hinused;
    delta.hlast = after.hlast;
    delta.hcur = after.hcur;
    delta.tcur = after.tcur;
    delta.current_gamma = after.current_gamma;
    return delta;
}

MlmgTelemetry operator- (const MlmgTelemetry& after, const MlmgTelemetry& before)
{
    MlmgTelemetry delta;
    delta.solve_calls = after.solve_calls - before.solve_calls;
    delta.iterations = after.iterations - before.iterations;
    delta.wall_seconds = after.wall_seconds - before.wall_seconds;
    delta.last_iterations = after.last_iterations;
    delta.last_residual = after.last_residual;
    return delta;
}

PhiSummary SummarizePhi (const MultiFab& phi)
{
    PhiSummary summary;
    summary.min = phi.min(0);
    summary.max = phi.max(0);
    summary.sum = phi.sum(0);
    summary.l1 = phi.norm1(0);
    summary.l2 = phi.norm2(0);
    summary.linf = phi.norm0(0);
    const auto npts = static_cast<Real>(phi.boxArray().numPts());
    summary.mean = npts > Real(0.0) ? summary.sum / npts : Real(0.0);
    summary.finite = std::isfinite(summary.min) && std::isfinite(summary.max) &&
                     std::isfinite(summary.sum) && std::isfinite(summary.l1) &&
                     std::isfinite(summary.l2) && std::isfinite(summary.linf);
    return summary;
}

void WriteSundialsStats (std::ostream& os, const SundialsIntegratorStats& stats)
{
    os << "{\"valid\":" << (stats.valid ? "true" : "false")
       << ",\"last_flag\":" << stats.last_flag
       << ",\"num_steps\":" << stats.num_steps
       << ",\"step_attempts\":" << stats.step_attempts
       << ",\"rhs_evals_0\":" << stats.rhs_evals_0
       << ",\"rhs_evals_1\":" << stats.rhs_evals_1
       << ",\"num_lin_solv_setups\":" << stats.num_lin_solv_setups
       << ",\"num_nonlin_solv_iters\":" << stats.num_nonlin_solv_iters
       << ",\"num_nonlin_solv_conv_fails\":" << stats.num_nonlin_solv_conv_fails
       << ",\"num_step_solve_fails\":" << stats.num_step_solve_fails
       << ",\"num_prec_evals\":" << stats.num_prec_evals
       << ",\"num_prec_solves\":" << stats.num_prec_solves
       << ",\"num_lin_iters\":" << stats.num_lin_iters
       << ",\"num_lin_conv_fails\":" << stats.num_lin_conv_fails
       << ",\"num_lin_rhs_evals\":" << stats.num_lin_rhs_evals
       << ",\"hinused\":" << stats.hinused
       << ",\"hlast\":" << stats.hlast
       << ",\"hcur\":" << stats.hcur
       << ",\"tcur\":" << stats.tcur
       << ",\"current_gamma\":" << stats.current_gamma
       << "}";
}

void WriteMlmgTelemetry (std::ostream& os, const MlmgTelemetry& stats)
{
    os << "{\"solve_calls\":" << stats.solve_calls
       << ",\"iterations\":" << stats.iterations
       << ",\"wall_seconds\":" << stats.wall_seconds
       << ",\"last_iterations\":" << stats.last_iterations
       << ",\"last_residual\":" << stats.last_residual
       << "}";
}

void WritePhiSummary (std::ostream& os, const PhiSummary& summary)
{
    os << "{\"min\":" << summary.min
       << ",\"max\":" << summary.max
       << ",\"sum\":" << summary.sum
       << ",\"mean\":" << summary.mean
       << ",\"l1\":" << summary.l1
       << ",\"l2\":" << summary.l2
       << ",\"linf\":" << summary.linf
       << ",\"finite\":" << (summary.finite ? "true" : "false")
       << "}";
}

} // namespace

int main (int argc, char* argv[])
{
    amrex::Initialize(argc,argv);

    main_main();

    amrex::Finalize();
    return 0;
}

void main_main ()
{

    if (AMREX_SPACEDIM != 2) {
        amrex::Abort("Only 2D supported; recompile with DIM=2");
    }
    
    // **********************************
    // SIMULATION PARAMETERS

    // number of cells on each side of the domain
    int n_cell;

    // size of each box (or grid)
    int max_grid_size;

    // total steps in simulation
    int nsteps;

    // how often to write a plotfile
    int plot_int;

    // time step
    Real dt;

    // use adaptive time step (dt used to set output times)
    bool adapt_dt = false;

    // adaptive time step relative and absolute tolerances
    Real reltol = 1.0e-4;
    Real abstol = 1.0e-9;

    // Advection and Diffusion Coefficients
    Real advCoeffx = 1.0;
    Real advCoeffy = 1.0;
    Real diffCoeffx = 1.0;
    Real diffCoeffy = 1.0;

    // MLMG settings for the SUNDIALS preconditioner solve
    int mlmg_max_iter = 100;
    int mlmg_max_fmg_iter = 0;
    int mlmg_verbose = 0;
    int mlmg_bottom_verbose = 0;
    bool mlmg_use_hypre = false;
    int mlmg_hypre_interface = 3;
    std::string mlmg_hypre_options_namespace = "hypre";
    Real mlmg_reltol = 1.e-10;
    Real mlmg_abstol = 0.0;

    bool rl_data_enabled = false;
    bool rl_skip_final_plotfiles = false;
    std::string rl_output_path = "advdiff_mlmg_rl_data.jsonl";
    std::string rl_schedule_path;
    RlAction rl_current_action;

    // inputs parameters
    {
        // ParmParse is way of reading inputs from the inputs file
        // pp.get means we require the inputs file to have it
        // pp.query means we optionally need the inputs file to have it - but we must supply a default here
        ParmParse pp;

        // We need to get n_cell from the inputs file - this is the number of cells on each side of
        //   a square (or cubic) domain.
        pp.get("n_cell",n_cell);

        // The domain is broken into boxes of size max_grid_size
        pp.get("max_grid_size",max_grid_size);

        // Default nsteps to 10, allow us to set it to something else in the inputs file
        nsteps = 10;
        pp.query("nsteps",nsteps);

        // Default plot_int to -1, allow us to set it to something else in the inputs file
        //  If plot_int < 0 then no plot files will be written
        plot_int = -1;
        pp.query("plot_int",plot_int);

        // time step
        pp.get("dt",dt);

        // use adaptive step sizes
        pp.query("adapt_dt",adapt_dt);

        // adaptive step tolerances
        pp.query("reltol",reltol);
        pp.query("abstol",abstol);

        pp.query("advCoeffx",advCoeffx);
        pp.query("advCoeffy",advCoeffy);
        pp.query("diffCoeffx",diffCoeffx);
        pp.query("diffCoeffy",diffCoeffy);

        ParmParse pp_mlmg("mlmg");
        pp_mlmg.query("max_iter",mlmg_max_iter);
        pp_mlmg.query("max_fmg_iter",mlmg_max_fmg_iter);
        pp_mlmg.query("verbose",mlmg_verbose);
        pp_mlmg.query("bottom_verbose",mlmg_bottom_verbose);
        pp_mlmg.query("use_hypre",mlmg_use_hypre);
        pp_mlmg.query("hypre_interface",mlmg_hypre_interface);
        pp_mlmg.query("hypre_options_namespace",mlmg_hypre_options_namespace);
        pp_mlmg.query("reltol",mlmg_reltol);
        pp_mlmg.query("abstol",mlmg_abstol);

        ParmParse pp_sundials("integration.sundials");
        pp_sundials.query("nlscoef", rl_current_action.nlscoef);
        pp_sundials.query("max_nonlinear_iters", rl_current_action.max_nonlinear_iters);
        pp_sundials.query("eps_lin", rl_current_action.eps_lin);
        pp_sundials.query("epsLin", rl_current_action.eps_lin);
        pp_sundials.query("lsetup_frequency", rl_current_action.lsetup_frequency);

        ParmParse pp_rl_data("rl_data");
        pp_rl_data.query("enabled", rl_data_enabled);
        pp_rl_data.query("output", rl_output_path);
        pp_rl_data.query("skip_final_plotfiles", rl_skip_final_plotfiles);

        ParmParse pp_rl_control("rl_control");
        pp_rl_control.query("schedule", rl_schedule_path);

#ifndef AMREX_USE_HYPRE
        if (mlmg_use_hypre) {
            amrex::Abort("mlmg.use_hypre requires AMReX to be built with HYPRE support");
        }
#endif
    }

    const std::vector<RlAction> rl_schedule = LoadRlSchedule(rl_schedule_path);
    std::size_t rl_next_schedule = 0;

    std::unique_ptr<std::ofstream> rl_output;
    if (rl_data_enabled && ParallelDescriptor::IOProcessor()) {
        rl_output = std::make_unique<std::ofstream>(rl_output_path);
        if (!(*rl_output)) {
            amrex::Abort("Could not open rl_data.output: " + rl_output_path);
        }
        rl_output->setf(std::ios::scientific);
        *rl_output << std::setprecision(17);
    }

    // **********************************
    // SIMULATION SETUP

    // AMREX_D_DECL means "do the first X of these, where X is the dimensionality of the simulation"
    IntVect dom_lo(AMREX_D_DECL(       0,        0,        0));
    IntVect dom_hi(AMREX_D_DECL(n_cell-1, n_cell-1, n_cell-1));

    // Make a single box that is the entire domain
    Box domain(dom_lo, dom_hi);

    // ba will contain a list of boxes that cover the domain
    // Initialize the boxarray "ba" from the single box "domain"
    BoxArray ba(domain);

    // Break up boxarray "ba" into chunks no larger than "max_grid_size" along a direction
    ba.maxSize(max_grid_size);

    // This defines the physical box, [0,1] in each direction.
    RealBox real_box({AMREX_D_DECL(-1.,-1.,-1.)},
                     {AMREX_D_DECL( 1., 1., 1.)});

    // periodic in all direction
    Array<int,AMREX_SPACEDIM> is_periodic{AMREX_D_DECL(1,1,1)};

    // geom contains information such as the physical domain size,
    //               number of points in the domain, and periodicity
    // This defines a Geometry object
    Geometry geom(domain, real_box, CoordSys::cartesian, is_periodic);

    // extract dx from the geometry object
    GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();
    GpuArray<Real,AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();
    GpuArray<Real,AMREX_SPACEDIM> prob_hi = geom.ProbHiArray();

    // Nghost = number of ghost cells for each array
    int Nghost = 1;

    // Ncomp = number of components for each array
    int Ncomp = 1;

    // How Boxes are distrubuted among MPI processes
    DistributionMapping dm(ba);

    // allocate phi MultiFab
    MultiFab phi(ba, dm, Ncomp, Nghost);

    // time = starting time in the simulation
    Real time = 0.0;

    // **********************************
    // INITIALIZE DATA

    InitializeData(phi,dx,prob_lo,prob_hi,time,advCoeffx,advCoeffy);

    // Write a plotfile of the initial data if plot_int > 0
    if (plot_int > 0)
    {
        int step = 0;
        const std::string& pltfile = amrex::Concatenate("plt",step,5);
        WriteSingleLevelPlotfile(pltfile, phi, {"phi"}, geom, time, 0);
    }

    auto rhs_function = [&](MultiFab& S_rhs, MultiFab& S_data, const Real /* time */) {

        // fill periodic ghost cells
        S_data.FillBoundary(geom.periodicity());

        S_rhs.setVal(0.);
        
        ComputeDiffusion(S_rhs, S_data, diffCoeffx, diffCoeffy, dx);
        ComputeAdvection(S_rhs, S_data, advCoeffx, advCoeffy, dx);
    };

    auto rhs_im_function = [&](MultiFab& S_rhs, MultiFab& S_data, const Real /* time */) {

        // fill periodic ghost cells
        S_data.FillBoundary(geom.periodicity());

        S_rhs.setVal(0.);
        
        ComputeDiffusion(S_rhs, S_data, diffCoeffx, diffCoeffy, dx);
    };

    auto rhs_ex_function = [&](MultiFab& S_rhs, MultiFab& S_data, const Real /* time */) {

        // fill periodic ghost cells
        S_data.FillBoundary(geom.periodicity());

        S_rhs.setVal(0.);
        
        ComputeAdvection(S_rhs, S_data, advCoeffx, advCoeffy, dx);
    };

    struct MLMGPreconditioner {
        std::unique_ptr<MLABecLaplacian> linop;
        std::unique_ptr<MLMG> solver;
    };

    std::unique_ptr<MLMGPreconditioner> mlmg_preconditioner;
    Real mlmg_gamma = std::numeric_limits<Real>::quiet_NaN();
    MlmgTelemetry mlmg_telemetry;

    auto build_mlmg_preconditioner = [&](Real gamma)
    {
        auto preconditioner = std::make_unique<MLMGPreconditioner>();

        LPInfo info;
        preconditioner->linop = std::make_unique<MLABecLaplacian>(
            Vector<Geometry>{geom}, Vector<BoxArray>{ba}, Vector<DistributionMapping>{dm}, info);
        preconditioner->linop->setMaxOrder(2);
        preconditioner->linop->setDomainBC(
            {AMREX_D_DECL(LinOpBCType::Periodic, LinOpBCType::Periodic, LinOpBCType::Periodic)},
            {AMREX_D_DECL(LinOpBCType::Periodic, LinOpBCType::Periodic, LinOpBCType::Periodic)});
        preconditioner->linop->setLevelBC(0, nullptr);
        preconditioner->linop->setScalars(1.0, 1.0);
        preconditioner->linop->setACoeffs(0, 1.0);

        std::array<MultiFab,AMREX_SPACEDIM> face_bcoef;
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            const BoxArray& face_ba = amrex::convert(ba, IntVect::TheDimensionVector(idim));
            face_bcoef[idim].define(face_ba, dm, 1, 0);
            const Real diff_coeff = (idim == 0) ? diffCoeffx : ((idim == 1) ? diffCoeffy : 0.0);
            face_bcoef[idim].setVal(gamma * diff_coeff);
        }
        preconditioner->linop->setBCoeffs(0, amrex::GetArrOfConstPtrs(face_bcoef));

        preconditioner->solver = std::make_unique<MLMG>(*preconditioner->linop);
        preconditioner->solver->setMaxIter(mlmg_max_iter);
        preconditioner->solver->setMaxFmgIter(mlmg_max_fmg_iter);
        preconditioner->solver->setVerbose(mlmg_verbose);
        preconditioner->solver->setBottomVerbose(mlmg_bottom_verbose);
#ifdef AMREX_USE_HYPRE
        if (mlmg_use_hypre) {
            Hypre::Interface hypre_interface = Hypre::Interface::ij;
            if (mlmg_hypre_interface == 1) {
                hypre_interface = Hypre::Interface::structed;
            } else if (mlmg_hypre_interface == 2) {
                hypre_interface = Hypre::Interface::semi_structed;
            } else if (mlmg_hypre_interface == 3) {
                hypre_interface = Hypre::Interface::ij;
            } else {
                amrex::Abort("mlmg.hypre_interface must be 1 (structed), 2 (semi_structed), or 3 (ij)");
            }
            preconditioner->solver->setBottomSolver(MLMG::BottomSolver::hypre);
            preconditioner->solver->setHypreInterface(hypre_interface);
            preconditioner->solver->setHypreOptionsNamespace(mlmg_hypre_options_namespace);
        }
#endif

        mlmg_preconditioner = std::move(preconditioner);
        mlmg_gamma = gamma;
    };

    auto precond_setup = [&](MultiFab& /* S_data */, MultiFab& /* S_rhs */, const Real /* time */,
                             bool jok, bool& jcur, const Real gamma)
    {
        const bool same_gamma = mlmg_preconditioner &&
            std::abs(gamma - mlmg_gamma) <=
            (10.0 * std::numeric_limits<Real>::epsilon() *
             std::max(Real(1.0), std::max(std::abs(gamma), std::abs(mlmg_gamma))));

        if (!jok) {
            build_mlmg_preconditioner(gamma);
            jcur = true;
        }
    };

    auto precond_solve = [&](MultiFab& S_soln, MultiFab& S_rhs, MultiFab& /* S_data */,
                             MultiFab& /* S_state_rhs */, const Real /* time */,
                             const Real /* gamma */, const Real /* delta */,
                             int /* lr */)
    {
        AMREX_ALWAYS_ASSERT(mlmg_preconditioner != nullptr);
        S_soln.setVal(0.0);
        const Real mlmg_solve_start = ParallelDescriptor::second();
        mlmg_preconditioner->solver->solve({&S_soln}, {&S_rhs}, mlmg_reltol, mlmg_abstol);
        Real mlmg_solve_time = ParallelDescriptor::second() - mlmg_solve_start;
        ParallelDescriptor::ReduceRealMax(mlmg_solve_time);

        const int mlmg_iterations = mlmg_preconditioner->solver->getNumIters();
        mlmg_telemetry.solve_calls += 1;
        mlmg_telemetry.iterations += mlmg_iterations;
        mlmg_telemetry.wall_seconds += mlmg_solve_time;
        mlmg_telemetry.last_iterations = mlmg_iterations;
        mlmg_telemetry.last_residual = mlmg_preconditioner->solver->getFinalResidual();
    };

    TimeIntegrator<MultiFab> integrator(phi, time);
    integrator.set_rhs(rhs_function);
    integrator.set_imex_rhs(rhs_im_function, rhs_ex_function);
    integrator.set_preconditioner(precond_setup, precond_solve);

    if (adapt_dt) {
        integrator.set_adaptive_step();
        integrator.set_tolerances(reltol, abstol);
    } else {
        integrator.set_time_step(dt);
    }

    if (rl_output) {
        const PhiSummary initial_phi = SummarizePhi(phi);
        *rl_output << "{\"event\":\"run_start\""
                   << ",\"n_cell\":" << n_cell
                   << ",\"max_grid_size\":" << max_grid_size
                   << ",\"nsteps\":" << nsteps
                   << ",\"dt\":" << dt
                   << ",\"adapt_dt\":" << (adapt_dt ? "true" : "false")
                   << ",\"advCoeffx\":" << advCoeffx
                   << ",\"advCoeffy\":" << advCoeffy
                   << ",\"diffCoeffx\":" << diffCoeffx
                   << ",\"diffCoeffy\":" << diffCoeffy
                   << ",\"initial_nlscoef\":" << rl_current_action.nlscoef
                   << ",\"initial_max_nonlinear_iters\":"
                   << rl_current_action.max_nonlinear_iters
                   << ",\"initial_eps_lin\":" << rl_current_action.eps_lin
                   << ",\"initial_lsetup_frequency\":"
                   << rl_current_action.lsetup_frequency
                   << ",\"schedule\":\"" << JsonEscape(rl_schedule_path) << "\""
                   << ",\"phi_initial\":";
        WritePhiSummary(*rl_output, initial_phi);
        *rl_output << "}\n";
    } else if (rl_data_enabled) {
        (void)SummarizePhi(phi);
    }

    Real evolution_start_time = ParallelDescriptor::second();
    int completed_steps = 0;
    bool episode_valid = true;

    for (int step = 1; step <= nsteps; ++step)
    {
        while (rl_next_schedule < rl_schedule.size() &&
               rl_schedule[rl_next_schedule].step <= step)
        {
            rl_current_action = rl_schedule[rl_next_schedule];
            ++rl_next_schedule;
        }

        if (integrator.supports_sundials_controls()) {
            integrator.set_sundials_nonlinear_control(
                {rl_current_action.nlscoef,
                 rl_current_action.max_nonlinear_iters,
                 rl_current_action.eps_lin,
                 rl_current_action.lsetup_frequency});
        }

        // Set time to evolve to
        const Real time_start = time;
        time += dt;

        PhiSummary phi_before;
        SundialsIntegratorStats sundials_before;
        MlmgTelemetry mlmg_before;
        if (rl_data_enabled) {
            phi_before = SummarizePhi(phi);
            sundials_before = integrator.get_sundials_stats();
            mlmg_before = mlmg_telemetry;
        }

        Real step_start_time = ParallelDescriptor::second();

        // Advance to output time
        integrator.evolve(phi, time);

        Real step_stop_time = ParallelDescriptor::second() - step_start_time;
        ParallelDescriptor::ReduceRealMax(step_stop_time);

        // Tell the I/O Processor to write out which step we're doing
        amrex::Print() << "Advanced step " << step << " in " << step_stop_time << " seconds; dt = " << dt << " time = " << time << "\n";

        if (rl_data_enabled) {
            const PhiSummary phi_after = SummarizePhi(phi);
            const SundialsIntegratorStats sundials_after = integrator.get_sundials_stats();
            const SundialsIntegratorStats sundials_delta = sundials_after - sundials_before;
            const MlmgTelemetry mlmg_delta = mlmg_telemetry - mlmg_before;
            const bool step_valid = phi_after.finite && sundials_after.last_flag >= 0;
            episode_valid = episode_valid && step_valid;

            if (rl_output) {
                *rl_output << "{\"event\":\"transition\""
                           << ",\"step\":" << step
                           << ",\"time_start\":" << time_start
                           << ",\"time_target\":" << time
                           << ",\"dt\":" << dt
                           << ",\"action\":{\"nlscoef\":" << rl_current_action.nlscoef
                           << ",\"max_nonlinear_iters\":"
                           << rl_current_action.max_nonlinear_iters
                           << ",\"eps_lin\":" << rl_current_action.eps_lin
                           << ",\"lsetup_frequency\":"
                           << rl_current_action.lsetup_frequency << "}"
                           << ",\"reward\":" << -step_stop_time
                           << ",\"evolve_wall_seconds\":" << step_stop_time
                           << ",\"valid\":" << (step_valid ? "true" : "false")
                           << ",\"phi_before\":";
                WritePhiSummary(*rl_output, phi_before);
                *rl_output << ",\"phi_after\":";
                WritePhiSummary(*rl_output, phi_after);
                *rl_output << ",\"sundials_before\":";
                WriteSundialsStats(*rl_output, sundials_before);
                *rl_output << ",\"sundials_after\":";
                WriteSundialsStats(*rl_output, sundials_after);
                *rl_output << ",\"sundials_delta\":";
                WriteSundialsStats(*rl_output, sundials_delta);
                *rl_output << ",\"mlmg_delta\":";
                WriteMlmgTelemetry(*rl_output, mlmg_delta);
                *rl_output << "}\n";
            }

            if (!step_valid) {
                amrex::Print() << "Stopping after invalid RL data step " << step
                               << "; SUNDIALS flag = " << sundials_after.last_flag
                               << ", finite phi = " << (phi_after.finite ? "true" : "false")
                               << "\n";
                break;
            }
        }

        completed_steps = step;

        // Write a plotfile of the current data (plot_int was defined in the inputs file)
        if (plot_int > 0 && step%plot_int == 0)
        {
            const std::string& pltfile = amrex::Concatenate("plt",step,5);
            WriteSingleLevelPlotfile(pltfile, phi, {"phi"}, geom, time, step);
        }
    }

    Real evolution_stop_time = ParallelDescriptor::second() - evolution_start_time;
    ParallelDescriptor::ReduceRealMax(evolution_stop_time);
    amrex::Print() << "Total evolution time = " << evolution_stop_time << " seconds\n";

    // exact solution
    BoxArray ba_exact(domain);
    DistributionMapping dm_exact(ba_exact);
    MultiFab phi_exact(ba_exact, dm_exact, Ncomp, n_cell);
    InitializeData(phi_exact,dx,prob_lo,prob_hi,time,advCoeffx,advCoeffy);
    phi_exact.SumBoundary(geom.periodicity());

    {
        const std::string& pltfile = amrex::Concatenate("exact",nsteps,5);
        if (!rl_skip_final_plotfiles) {
            WriteSingleLevelPlotfile(pltfile, phi_exact, {"phi"}, geom, time, nsteps);
        }
    }

    MultiFab phi_exact_dist(ba,dm,Ncomp,0);
    phi_exact_dist.ParallelCopy(phi_exact,0,0,1);
    
    MultiFab::Subtract(phi_exact_dist,phi,0,0,1,0);

    {
        const std::string& pltfile = amrex::Concatenate("diff",nsteps,5);
        if (!rl_skip_final_plotfiles) {
            WriteSingleLevelPlotfile(pltfile, phi_exact_dist, {"phi"}, geom, time, nsteps);
        }
    }

    Real error = phi_exact_dist.norm1(0,geom.periodicity());
    amrex::Print() << "L1 error = " << error << std::endl;

    if (rl_data_enabled) {
        const PhiSummary final_phi = SummarizePhi(phi);
        episode_valid = episode_valid && final_phi.finite && std::isfinite(error);
        if (rl_output) {
            *rl_output << "{\"event\":\"run_end\""
                       << ",\"completed_steps\":" << completed_steps
                       << ",\"requested_steps\":" << nsteps
                       << ",\"valid\":" << (episode_valid ? "true" : "false")
                       << ",\"total_evolution_time\":" << evolution_stop_time
                       << ",\"l1_error\":" << error
                       << ",\"mlmg_total\":";
            WriteMlmgTelemetry(*rl_output, mlmg_telemetry);
            *rl_output << ",\"phi_final\":";
            WritePhiSummary(*rl_output, final_phi);
            *rl_output << "}\n";
        }
    }
}

void InitializeData(MultiFab& phi,
                    const GpuArray<Real,AMREX_SPACEDIM> dx,
                    const GpuArray<Real,AMREX_SPACEDIM> prob_lo,
                    const GpuArray<Real,AMREX_SPACEDIM> prob_hi,
                    const Real& time,
                    const Real& Ax,
                    const Real& Ay) {

    int ng = phi.nGrow();
    
    GpuArray<Real,AMREX_SPACEDIM> L;
    for (int d=0; d<AMREX_SPACEDIM; ++d) {
        L[d] = prob_hi[d] - prob_lo[d];
    }

    // loop over boxes
    for (MFIter mfi(phi); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.growntilebox(ng);
        const Array4<Real>& phi_array = phi.array(mfi);

        Real sigma = 0.1;
        Real a = 1.0/(sigma*sqrt(2*M_PI));
        Real b = -0.5/(sigma*sigma);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            Real x = prob_lo[0] + (((Real) i) + 0.5) * dx[0];
            Real y = prob_lo[1] + (((Real) j) + 0.5) * dx[1];
            Real r = (x-Ax*time) * (x-Ax*time) + (y-Ay*time) * (y-Ay*time);
            phi_array(i,j,k) = a * std::exp(b * r);
        });
    }

}

void ComputeDiffusion(MultiFab& S_rhs,
                      MultiFab& S_data,
                      const Real& Dx,
                      const Real& Dy,
                      const GpuArray<Real,AMREX_SPACEDIM> dx) {

    for ( MFIter mfi(S_data,TilingIfNotGPU()); mfi.isValid(); ++mfi )
    {
        const Box& bx = mfi.tilebox();

        const Array4<const Real>& phi_array = S_data.array(mfi);
        const Array4<      Real>& rhs_array = S_rhs.array(mfi);

        // fill the right-hand-side for phi
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            rhs_array(i,j,k) += Dx * ( (phi_array(i+1,j,k) - 2.*phi_array(i,j,k) + phi_array(i-1,j,k)) / (dx[0]*dx[0]) )
                              + Dy * ( (phi_array(i,j+1,k) - 2.*phi_array(i,j,k) + phi_array(i,j-1,k)) / (dx[1]*dx[1]) );
        });
    }
}


void ComputeAdvection(MultiFab& S_rhs,
                      MultiFab& S_data,
                      const Real& Ax,
                      const Real& Ay,
                      const GpuArray<Real,AMREX_SPACEDIM> dx) {

    Real dxInv = 1.0 / dx[0];
    Real dyInv = 1.0 / dx[1];
    Real sideCoeffx = Ax * dxInv;
    Real sideCoeffy = Ay * dyInv;

    for (MFIter mfi(S_data,TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.tilebox();

        const Array4<const Real>& phi_array = S_data.array(mfi);
        const Array4<      Real>& rhs_array = S_rhs.array(mfi);

        // x-direction
        if (Ax > 0)
        {
            ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                rhs_array(i,j,k) -= sideCoeffx * (phi_array(i,j,k) - phi_array(i-1,j,k));
            });
        }
        else
        {
            ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                rhs_array(i,j,k) -= sideCoeffx * (phi_array(i+1,j,k) - phi_array(i,j,k));
            });
        }

        // y-direction
        if (Ay > 0)
        {
            ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                rhs_array(i,j,k) -= sideCoeffy * (phi_array(i,j,k) - phi_array(i,j-1,k));
            });
        }
        else
        {
            ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                rhs_array(i,j,k) -= sideCoeffy * (phi_array(i,j+1,k) - phi_array(i,j,k));
            });
        }
    }
}
