#ifndef RL_TRACKING_H_
#define RL_TRACKING_H_

#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_TimeIntegrator.H>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <memory>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

namespace AdvDiffRl {

struct Action {
    int step = 0;
    amrex::Real nlscoef = amrex::Real(-1.0);
    int max_nonlinear_iters = 0;
    amrex::Real eps_lin = amrex::Real(-1.0);
    int lsetup_frequency = 0;
};

struct MlmgTelemetry {
    long solve_calls = 0;
    long iterations = 0;
    amrex::Real wall_seconds = amrex::Real(0.0);
    int last_iterations = 0;
    amrex::Real last_residual = amrex::Real(-1.0);
};

struct PhiSummary {
    amrex::Real min = amrex::Real(0.0);
    amrex::Real max = amrex::Real(0.0);
    amrex::Real sum = amrex::Real(0.0);
    amrex::Real mean = amrex::Real(0.0);
    amrex::Real l1 = amrex::Real(0.0);
    amrex::Real l2 = amrex::Real(0.0);
    amrex::Real linf = amrex::Real(0.0);
    bool finite = true;
};

struct StepSnapshot {
    PhiSummary phi;
    amrex::SundialsIntegratorStats sundials;
    MlmgTelemetry mlmg;
};

struct RunMetadata {
    int n_cell = 0;
    int max_grid_size = 0;
    int nsteps = 0;
    amrex::Real dt = amrex::Real(0.0);
    bool adapt_dt = false;
    amrex::Real advCoeffx = amrex::Real(0.0);
    amrex::Real advCoeffy = amrex::Real(0.0);
    amrex::Real diffCoeffx = amrex::Real(0.0);
    amrex::Real diffCoeffy = amrex::Real(0.0);
    std::string schedule_path;
};

inline std::string Trim (std::string value)
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

inline std::string JsonEscape (const std::string& value)
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

inline std::vector<Action> LoadSchedule (const std::string& path)
{
    std::vector<Action> schedule;
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

        Action action;
        action.step = std::stoi(fields[0]);
        action.nlscoef = static_cast<amrex::Real>(std::stod(fields[1]));
        action.max_nonlinear_iters = std::stoi(fields[2]);
        if (fields.size() == 5) {
            action.eps_lin = static_cast<amrex::Real>(std::stod(fields[3]));
            action.lsetup_frequency = std::stoi(fields[4]);
        }
        schedule.push_back(action);
    }

    std::sort(schedule.begin(), schedule.end(),
              [](const Action& a, const Action& b) { return a.step < b.step; });
    return schedule;
}

inline void ApplyScheduledAction (const std::vector<Action>& schedule, std::size_t& next_schedule,
                                  int step, Action& action)
{
    while (next_schedule < schedule.size() && schedule[next_schedule].step <= step) {
        action = schedule[next_schedule];
        ++next_schedule;
    }
}

inline std::unique_ptr<std::ofstream> OpenOutputFile (bool enabled, const std::string& path)
{
    std::unique_ptr<std::ofstream> output;
    if (enabled && amrex::ParallelDescriptor::IOProcessor()) {
        output = std::make_unique<std::ofstream>(path);
        if (!(*output)) {
            amrex::Abort("Could not open rl_data.output: " + path);
        }
        output->setf(std::ios::scientific);
        *output << std::setprecision(17);
    }
    return output;
}

inline amrex::SundialsIntegratorStats operator- (const amrex::SundialsIntegratorStats& after,
                                                 const amrex::SundialsIntegratorStats& before)
{
    amrex::SundialsIntegratorStats delta;
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

inline MlmgTelemetry operator- (const MlmgTelemetry& after, const MlmgTelemetry& before)
{
    MlmgTelemetry delta;
    delta.solve_calls = after.solve_calls - before.solve_calls;
    delta.iterations = after.iterations - before.iterations;
    delta.wall_seconds = after.wall_seconds - before.wall_seconds;
    delta.last_iterations = after.last_iterations;
    delta.last_residual = after.last_residual;
    return delta;
}

inline void RecordMlmgSolve (MlmgTelemetry& telemetry, int iterations,
                             amrex::Real wall_seconds, amrex::Real residual)
{
    telemetry.solve_calls += 1;
    telemetry.iterations += iterations;
    telemetry.wall_seconds += wall_seconds;
    telemetry.last_iterations = iterations;
    telemetry.last_residual = residual;
}

inline PhiSummary SummarizePhi (const amrex::MultiFab& phi)
{
    PhiSummary summary;
    summary.min = phi.min(0);
    summary.max = phi.max(0);
    summary.sum = phi.sum(0);
    summary.l1 = phi.norm1(0);
    summary.l2 = phi.norm2(0);
    summary.linf = phi.norm0(0);
    const auto npts = static_cast<amrex::Real>(phi.boxArray().numPts());
    summary.mean = npts > amrex::Real(0.0) ? summary.sum / npts : amrex::Real(0.0);
    summary.finite = std::isfinite(summary.min) && std::isfinite(summary.max) &&
                     std::isfinite(summary.sum) && std::isfinite(summary.l1) &&
                     std::isfinite(summary.l2) && std::isfinite(summary.linf);
    return summary;
}

inline StepSnapshot CaptureStepSnapshot (const amrex::MultiFab& phi,
                                         const amrex::SundialsIntegratorStats& sundials,
                                         const MlmgTelemetry& mlmg)
{
    return {SummarizePhi(phi), sundials, mlmg};
}

inline bool StepIsValid (const StepSnapshot& snapshot)
{
    return snapshot.phi.finite && snapshot.sundials.last_flag >= 0;
}

inline void WriteSundialsStats (std::ostream& os,
                                const amrex::SundialsIntegratorStats& stats)
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

inline void WriteMlmgTelemetry (std::ostream& os, const MlmgTelemetry& stats)
{
    os << "{\"solve_calls\":" << stats.solve_calls
       << ",\"iterations\":" << stats.iterations
       << ",\"wall_seconds\":" << stats.wall_seconds
       << ",\"last_iterations\":" << stats.last_iterations
       << ",\"last_residual\":" << stats.last_residual
       << "}";
}

inline void WritePhiSummary (std::ostream& os, const PhiSummary& summary)
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

inline void RecordRunStart (std::ostream* output, const amrex::MultiFab& phi,
                            const RunMetadata& metadata, const Action& initial_action,
                            bool enabled)
{
    if (!enabled) {
        return;
    }

    const PhiSummary initial_phi = SummarizePhi(phi);
    if (output == nullptr) {
        return;
    }

    *output << "{\"event\":\"run_start\""
            << ",\"n_cell\":" << metadata.n_cell
            << ",\"max_grid_size\":" << metadata.max_grid_size
            << ",\"nsteps\":" << metadata.nsteps
            << ",\"dt\":" << metadata.dt
            << ",\"adapt_dt\":" << (metadata.adapt_dt ? "true" : "false")
            << ",\"advCoeffx\":" << metadata.advCoeffx
            << ",\"advCoeffy\":" << metadata.advCoeffy
            << ",\"diffCoeffx\":" << metadata.diffCoeffx
            << ",\"diffCoeffy\":" << metadata.diffCoeffy
            << ",\"initial_nlscoef\":" << initial_action.nlscoef
            << ",\"initial_max_nonlinear_iters\":"
            << initial_action.max_nonlinear_iters
            << ",\"initial_eps_lin\":" << initial_action.eps_lin
            << ",\"initial_lsetup_frequency\":"
            << initial_action.lsetup_frequency
            << ",\"schedule\":\"" << JsonEscape(metadata.schedule_path) << "\""
            << ",\"phi_initial\":";
    WritePhiSummary(*output, initial_phi);
    *output << "}\n";
}

inline void WriteTransition (std::ostream& output, int step, amrex::Real time_start,
                             amrex::Real time_target, amrex::Real dt, const Action& action,
                             amrex::Real step_wall_seconds, const StepSnapshot& before,
                             const StepSnapshot& after)
{
    const auto sundials_delta = after.sundials - before.sundials;
    const auto mlmg_delta = after.mlmg - before.mlmg;
    const bool step_valid = StepIsValid(after);

    output << "{\"event\":\"transition\""
           << ",\"step\":" << step
           << ",\"time_start\":" << time_start
           << ",\"time_target\":" << time_target
           << ",\"dt\":" << dt
           << ",\"action\":{\"nlscoef\":" << action.nlscoef
           << ",\"max_nonlinear_iters\":"
           << action.max_nonlinear_iters
           << ",\"eps_lin\":" << action.eps_lin
           << ",\"lsetup_frequency\":"
           << action.lsetup_frequency << "}"
           << ",\"reward\":" << -step_wall_seconds
           << ",\"evolve_wall_seconds\":" << step_wall_seconds
           << ",\"valid\":" << (step_valid ? "true" : "false")
           << ",\"phi_before\":";
    WritePhiSummary(output, before.phi);
    output << ",\"phi_after\":";
    WritePhiSummary(output, after.phi);
    output << ",\"sundials_before\":";
    WriteSundialsStats(output, before.sundials);
    output << ",\"sundials_after\":";
    WriteSundialsStats(output, after.sundials);
    output << ",\"sundials_delta\":";
    WriteSundialsStats(output, sundials_delta);
    output << ",\"mlmg_delta\":";
    WriteMlmgTelemetry(output, mlmg_delta);
    output << "}\n";
}

inline bool RecordTransition (std::ostream* output, int step, amrex::Real time_start,
                              amrex::Real time_target, amrex::Real dt, const Action& action,
                              amrex::Real step_wall_seconds, const StepSnapshot& before,
                              const StepSnapshot& after)
{
    if (output != nullptr) {
        WriteTransition(*output, step, time_start, time_target, dt, action,
                        step_wall_seconds, before, after);
    }
    return StepIsValid(after);
}

inline bool RecordRunEnd (std::ostream* output, const amrex::MultiFab& phi, int completed_steps,
                          int requested_steps, bool episode_valid,
                          amrex::Real evolution_wall_seconds, amrex::Real l1_error,
                          const MlmgTelemetry& mlmg)
{
    const PhiSummary final_phi = SummarizePhi(phi);
    const bool valid = episode_valid && final_phi.finite && std::isfinite(l1_error);

    if (output != nullptr) {
        *output << "{\"event\":\"run_end\""
                << ",\"completed_steps\":" << completed_steps
                << ",\"requested_steps\":" << requested_steps
                << ",\"valid\":" << (valid ? "true" : "false")
                << ",\"total_evolution_time\":" << evolution_wall_seconds
                << ",\"l1_error\":" << l1_error
                << ",\"mlmg_total\":";
        WriteMlmgTelemetry(*output, mlmg);
        *output << ",\"phi_final\":";
        WritePhiSummary(*output, final_phi);
        *output << "}\n";
    }

    return valid;
}

} // namespace AdvDiffRl

#endif
