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
#include "rl_tracking.H"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cmath>
#include <limits>
#include <memory>
#include <string>

using namespace amrex;
namespace rl = AdvDiffRl;

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
    std::string rl_control_mode = "schedule";
    rl::Action rl_current_action;

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
        pp_sundials.query("jac_eval_frequency", rl_current_action.jac_eval_frequency);
        pp_sundials.query("jacevalfrequency", rl_current_action.jac_eval_frequency);

        ParmParse pp_rl_data("rl_data");
        pp_rl_data.query("enabled", rl_data_enabled);
        pp_rl_data.query("output", rl_output_path);
        pp_rl_data.query("skip_final_plotfiles", rl_skip_final_plotfiles);

        ParmParse pp_rl_control("rl_control");
        pp_rl_control.query("schedule", rl_schedule_path);
        pp_rl_control.query("mode", rl_control_mode);

#ifndef AMREX_USE_HYPRE
        if (mlmg_use_hypre) {
            amrex::Abort("mlmg.use_hypre requires AMReX to be built with HYPRE support");
        }
#endif
    }

    const bool rl_stdin_json_control = (rl_control_mode == "stdin_json");
    if (rl_control_mode != "schedule" && rl_control_mode != "stdin_json") {
        amrex::Abort("rl_control.mode must be 'schedule' or 'stdin_json'");
    }
    if (rl_stdin_json_control && !rl_schedule_path.empty()) {
        amrex::Abort("rl_control.schedule cannot be combined with rl_control.mode=stdin_json");
    }

    const auto rl_schedule = rl::LoadSchedule(rl_schedule_path);
    std::size_t rl_next_schedule = 0;
    auto rl_output = rl::OpenOutputFile(rl_data_enabled, rl_output_path);

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
        std::array<MultiFab,AMREX_SPACEDIM> face_bcoef;
    };

    std::unique_ptr<MLMGPreconditioner> mlmg_preconditioner;
    Real mlmg_gamma = std::numeric_limits<Real>::quiet_NaN();
    rl::MlmgTelemetry mlmg_telemetry;

    auto gamma_has_changed = [&] (Real gamma)
    {
        if (!std::isfinite(mlmg_gamma)) {
            return true;
        }
        const Real scale = std::max({Real(1.0), std::abs(gamma), std::abs(mlmg_gamma)});
        return std::abs(gamma - mlmg_gamma) >
               Real(10.0) * std::numeric_limits<Real>::epsilon() * scale;
    };

    auto update_mlmg_gamma = [&](MLMGPreconditioner& preconditioner, Real gamma)
    {
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            const Real diff_coeff = (idim == 0) ? diffCoeffx : ((idim == 1) ? diffCoeffy : 0.0);
            preconditioner.face_bcoef[idim].setVal(gamma * diff_coeff);
        }
        preconditioner.linop->setBCoeffs(0, amrex::GetArrOfConstPtrs(preconditioner.face_bcoef));
        mlmg_gamma = gamma;
    };

    auto gamma_has_changed = [&] (Real gamma)
    {
        if (!std::isfinite(mlmg_gamma)) {
            return true;
        }
        const Real scale = std::max({Real(1.0), std::abs(gamma), std::abs(mlmg_gamma)});
        return std::abs(gamma - mlmg_gamma) >
               Real(10.0) * std::numeric_limits<Real>::epsilon() * scale;
    };

    auto update_mlmg_gamma = [&](MLMGPreconditioner& preconditioner, Real gamma)
    {
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            const Real diff_coeff = (idim == 0) ? diffCoeffx : ((idim == 1) ? diffCoeffy : 0.0);
            preconditioner.face_bcoef[idim].setVal(gamma * diff_coeff);
        }
        preconditioner.linop->setBCoeffs(0, amrex::GetArrOfConstPtrs(preconditioner.face_bcoef));
        mlmg_gamma = gamma;
    };

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

        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            const BoxArray& face_ba = amrex::convert(ba, IntVect::TheDimensionVector(idim));
            preconditioner->face_bcoef[idim].define(face_ba, dm, 1, 0);
        }
        update_mlmg_gamma(*preconditioner, gamma);

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
        // The implicit diffusion Jacobian is constant for this problem, so a
        // SUNDIALS refresh only needs to update the gamma-scaled operator.
        if (mlmg_preconditioner == nullptr) {
            build_mlmg_preconditioner(gamma);
            jcur = true;
            return;
        }

        if (!jok) {
            if (gamma_has_changed(gamma)) {
                update_mlmg_gamma(*mlmg_preconditioner, gamma);
            }
            jcur = true;
        } else if (gamma_has_changed(gamma)) {
            update_mlmg_gamma(*mlmg_preconditioner, gamma);
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
        rl::RecordMlmgSolve(mlmg_telemetry, mlmg_iterations, mlmg_solve_time,
                            mlmg_preconditioner->solver->getFinalResidual());
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

    const rl::RunMetadata rl_metadata{n_cell, max_grid_size, nsteps, dt, adapt_dt,
                                      advCoeffx, advCoeffy, diffCoeffx, diffCoeffy,
                                      rl_schedule_path, rl_control_mode};
    rl::RecordRunStart(rl_output.get(), phi, rl_metadata, rl_current_action, rl_data_enabled);
    if (rl_stdin_json_control && amrex::ParallelDescriptor::IOProcessor()) {
        std::cout.setf(std::ios::scientific);
        std::cout << std::setprecision(17);
        rl::WriteRunStart(std::cout, phi, rl_metadata, rl_current_action);
        std::cout.flush();
    }

    Real evolution_start_time = ParallelDescriptor::second();
    int completed_steps = 0;
    bool episode_valid = true;
    const bool rl_capture_enabled = rl_data_enabled || rl_stdin_json_control;

    for (int step = 1; step <= nsteps; ++step)
    {
        // Set time to evolve to
        const Real time_start = time;
        const Real time_target = time_start + dt;

        rl::StepSnapshot rl_before;
        if (rl_capture_enabled) {
            rl_before = rl::CaptureStepSnapshot(phi, integrator.get_sundials_stats(),
                                                mlmg_telemetry);
        }

        if (rl_stdin_json_control) {
            if (amrex::ParallelDescriptor::IOProcessor()) {
                rl::WriteControlRequest(std::cout, step, time_start, time_target, dt,
                                        rl_current_action, rl_before, rl_metadata);
                std::cout.flush();
            }
            rl::ReadJsonActionFromStdin(step, rl_current_action);
        } else {
            rl::ApplyScheduledAction(rl_schedule, rl_next_schedule, step, rl_current_action);
        }

        if (integrator.supports_sundials_controls()) {
            integrator.set_sundials_nonlinear_control(
                {rl_current_action.nlscoef,
                 rl_current_action.max_nonlinear_iters,
                 rl_current_action.eps_lin,
                 rl_current_action.lsetup_frequency,
                 rl_current_action.jac_eval_frequency});
        }

        time = time_target;
        Real step_start_time = ParallelDescriptor::second();

        // Advance to output time
        integrator.evolve(phi, time);

        Real step_stop_time = ParallelDescriptor::second() - step_start_time;
        ParallelDescriptor::ReduceRealMax(step_stop_time);

        // Tell the I/O Processor to write out which step we're doing
        amrex::Print() << "Advanced step " << step << " in " << step_stop_time << " seconds; dt = " << dt << " time = " << time << "\n";

        if (rl_capture_enabled) {
            const auto rl_after = rl::CaptureStepSnapshot(phi, integrator.get_sundials_stats(),
                                                          mlmg_telemetry);
            bool step_valid = rl::StepIsValid(rl_after);
            if (rl_data_enabled) {
                step_valid = rl::RecordTransition(
                    rl_output.get(), step, time_start, time, dt, rl_current_action,
                    step_stop_time, rl_before, rl_after);
            }
            if (rl_stdin_json_control && amrex::ParallelDescriptor::IOProcessor()) {
                rl::WriteTransition(std::cout, step, time_start, time, dt,
                                    rl_current_action, step_stop_time, rl_before, rl_after);
                std::cout.flush();
            }
            episode_valid = episode_valid && step_valid;

            if (!step_valid) {
                amrex::Print() << "Stopping after invalid RL data step " << step
                               << "; SUNDIALS flag = " << rl_after.sundials.last_flag
                               << ", finite phi = " << (rl_after.phi.finite ? "true" : "false")
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
        episode_valid = rl::RecordRunEnd(rl_output.get(), phi, completed_steps, nsteps,
                                         episode_valid, evolution_stop_time, error,
                                         mlmg_telemetry);
    }
    if (rl_stdin_json_control && amrex::ParallelDescriptor::IOProcessor()) {
        episode_valid = rl::WriteRunEnd(std::cout, phi, completed_steps, nsteps,
                                        episode_valid, evolution_stop_time, error,
                                        mlmg_telemetry);
        std::cout.flush();
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
