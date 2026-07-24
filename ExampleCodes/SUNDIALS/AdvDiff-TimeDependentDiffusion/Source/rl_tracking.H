#ifndef RL_TRACKING_H_
#define RL_TRACKING_H_

#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_TimeIntegrator.H>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
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
    long jac_eval_frequency = 0;
    int force_lsetup = 0;
    int force_jac_eval = 0;
};

struct HypreAction {
    int setup = 0;
    amrex::Real strong_threshold = amrex::Real(-1.0);
    int coarsen_type = -1;
    int interp_type = -1;
    amrex::Real max_row_sum = amrex::Real(-1.0);
    int relax_type = -1;
    int num_sweeps = -1;
    int cycle_type = -1;
    int max_levels = -1;
    amrex::Real trunc_factor = amrex::Real(-1.0);
    int pmax_elmts = -1;
    int agg_num_levels = -1;
    int agg_interp_type = -1;
    amrex::Real agg_trunc_factor = amrex::Real(-1.0);
    int agg_pmax_elmts = -1;
    amrex::Real relax_wt = amrex::Real(-1.0);
    int relax_order = -1;
    int max_coarse_size = -1;
};

struct HypreSetupContext {
    int setup = 0;
    amrex::Real time = amrex::Real(0.0);
    amrex::Real gamma = amrex::Real(0.0);
    amrex::Real dt = amrex::Real(0.0);
    int n_cell = 0;
    int max_grid_size = 0;
    int dimension = AMREX_SPACEDIM;
    amrex::Real dx = amrex::Real(0.0);
    amrex::Real advCoeffx = amrex::Real(0.0);
    amrex::Real advCoeffy = amrex::Real(0.0);
    amrex::Real active_diffCoeffx = amrex::Real(0.0);
    amrex::Real active_diffCoeffy = amrex::Real(0.0);
    amrex::Real cfl_adv_x = amrex::Real(0.0);
    amrex::Real cfl_adv_y = amrex::Real(0.0);
    amrex::Real cfl_diff_x = amrex::Real(0.0);
    amrex::Real cfl_diff_y = amrex::Real(0.0);
    amrex::Real tol = amrex::Real(0.0);
    int max_iter = 0;
};

struct MlmgTelemetry {
    long precond_setup_calls = 0;
    amrex::Real precond_setup_wall_seconds = amrex::Real(0.0);
    long solve_calls = 0;
    long iterations = 0;
    amrex::Real wall_seconds = amrex::Real(0.0);
    long hypre_setup_calls = 0;
    long hypre_solve_calls = 0;
    amrex::Real hypre_setup_wall_seconds = amrex::Real(0.0);
    amrex::Real hypre_solve_wall_seconds = amrex::Real(0.0);
    amrex::Real last_precond_setup_wall_seconds = amrex::Real(0.0);
    int last_iterations = 0;
    amrex::Real last_residual = amrex::Real(-1.0);
    amrex::Real last_hypre_setup_wall_seconds = amrex::Real(0.0);
    amrex::Real last_hypre_solve_wall_seconds = amrex::Real(0.0);
    amrex::Real last_hypre_cum_nnz_ap = amrex::Real(-1.0);
    int last_hypre_hierarchy_levels = 0;
    long last_hypre_finest_rows = 0;
    long last_hypre_coarsest_rows = 0;
};

inline bool operator== (const HypreAction& lhs, const HypreAction& rhs)
{
    return lhs.strong_threshold == rhs.strong_threshold &&
           lhs.coarsen_type == rhs.coarsen_type &&
           lhs.interp_type == rhs.interp_type &&
           lhs.max_row_sum == rhs.max_row_sum &&
           lhs.relax_type == rhs.relax_type &&
           lhs.num_sweeps == rhs.num_sweeps &&
           lhs.cycle_type == rhs.cycle_type &&
           lhs.max_levels == rhs.max_levels &&
           lhs.trunc_factor == rhs.trunc_factor &&
           lhs.pmax_elmts == rhs.pmax_elmts &&
           lhs.agg_num_levels == rhs.agg_num_levels &&
           lhs.agg_interp_type == rhs.agg_interp_type &&
           lhs.agg_trunc_factor == rhs.agg_trunc_factor &&
           lhs.agg_pmax_elmts == rhs.agg_pmax_elmts &&
           lhs.relax_wt == rhs.relax_wt &&
           lhs.relax_order == rhs.relax_order &&
           lhs.max_coarse_size == rhs.max_coarse_size;
}

inline bool operator!= (const HypreAction& lhs, const HypreAction& rhs)
{
    return !(lhs == rhs);
}

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

struct ActiveDiffusion {
    int segment = 0;
    amrex::Real x_multiplier = amrex::Real(1.0);
    amrex::Real y_multiplier = amrex::Real(1.0);
    amrex::Real x = amrex::Real(0.0);
    amrex::Real y = amrex::Real(0.0);
};

struct RunMetadata {
    int n_cell = 0;
    int max_grid_size = 0;
    int max_agent_steps = 0;
    amrex::Real final_time = amrex::Real(0.0);
    amrex::Real dt_nominal = amrex::Real(0.0);
    bool adapt_dt = false;
    amrex::Real advCoeffx = amrex::Real(0.0);
    amrex::Real advCoeffy = amrex::Real(0.0);
    amrex::Real diffCoeffx = amrex::Real(0.0);
    amrex::Real diffCoeffy = amrex::Real(0.0);
    std::vector<amrex::Real> diffusion_times;
    std::vector<amrex::Real> diffusion_x_multipliers;
    std::vector<amrex::Real> diffusion_y_multipliers;
    std::string diffusion_interpolation = "constant";
    bool diffusion_reinitialize_at_jumps = true;
    std::string schedule_path;
    std::string control_mode;
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

        if (fields.size() != 6 && fields.size() != 8) {
            amrex::Abort("Bad rl_control.schedule line " + std::to_string(line_number) +
                         ": expected step,nlscoef,max_nonlinear_iters,"
                         "eps_lin,lsetup_frequency,jac_eval_frequency"
                         "[,force_lsetup,force_jac_eval]");
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
        action.eps_lin = static_cast<amrex::Real>(std::stod(fields[3]));
        action.lsetup_frequency = std::stoi(fields[4]);
        action.jac_eval_frequency = std::stol(fields[5]);
        if (fields.size() == 8) {
            action.force_lsetup = std::stoi(fields[6]);
            action.force_jac_eval = std::stoi(fields[7]);
        }
        schedule.push_back(action);
    }

    std::sort(schedule.begin(), schedule.end(),
              [](const Action& a, const Action& b) { return a.step < b.step; });
    return schedule;
}

inline void ApplyScheduledAction (const std::vector<Action>& schedule, std::size_t& next_schedule,
                                  int control_step, Action& action)
{
    while (next_schedule < schedule.size() && schedule[next_schedule].step <= control_step) {
        action = schedule[next_schedule];
        ++next_schedule;
    }
}

inline bool ExtractJsonNumber (const std::string& text, const std::string& key,
                               double& value)
{
    const std::string quoted_key = "\"" + key + "\"";
    const auto key_pos = text.find(quoted_key);
    if (key_pos == std::string::npos) {
        return false;
    }

    const auto colon_pos = text.find(':', key_pos + quoted_key.size());
    if (colon_pos == std::string::npos) {
        return false;
    }

    auto number_pos = colon_pos + 1;
    while (number_pos < text.size() &&
           std::isspace(static_cast<unsigned char>(text[number_pos])))
    {
        ++number_pos;
    }

    char* end = nullptr;
    const char* start = text.c_str() + number_pos;
    value = std::strtod(start, &end);
    return end != start;
}

inline bool ParseJsonAction (const std::string& line, int step, Action& action)
{
    Action parsed = action;
    parsed.step = step;

    double value = 0.0;
    bool saw_field = false;
    if (ExtractJsonNumber(line, "nlscoef", value)) {
        parsed.nlscoef = static_cast<amrex::Real>(value);
        saw_field = true;
    }
    if (ExtractJsonNumber(line, "max_nonlinear_iters", value)) {
        parsed.max_nonlinear_iters = static_cast<int>(value);
        saw_field = true;
    }
    if (ExtractJsonNumber(line, "eps_lin", value) || ExtractJsonNumber(line, "epsLin", value)) {
        parsed.eps_lin = static_cast<amrex::Real>(value);
        saw_field = true;
    }
    if (ExtractJsonNumber(line, "lsetup_frequency", value)) {
        parsed.lsetup_frequency = static_cast<int>(value);
        saw_field = true;
    }
    if (ExtractJsonNumber(line, "jac_eval_frequency", value) ||
        ExtractJsonNumber(line, "jacevalfrequency", value))
    {
        parsed.jac_eval_frequency = static_cast<long>(value);
        saw_field = true;
    }
    if (ExtractJsonNumber(line, "force_lsetup", value)) {
        parsed.force_lsetup = static_cast<int>(value);
        saw_field = true;
    }
    if (ExtractJsonNumber(line, "force_jac_eval", value) ||
        ExtractJsonNumber(line, "force_jaceval", value))
    {
        parsed.force_jac_eval = static_cast<int>(value);
        saw_field = true;
    }

    if (saw_field) {
        action = parsed;
    }
    return saw_field;
}

inline bool ParseJsonHypreAction (const std::string& line, int setup, HypreAction& action)
{
    HypreAction parsed = action;
    parsed.setup = setup;

    double value = 0.0;
    bool saw_field = false;
    if (ExtractJsonNumber(line, "strong_threshold", value)) {
        parsed.strong_threshold = static_cast<amrex::Real>(value);
        saw_field = true;
    }
    if (ExtractJsonNumber(line, "coarsen_type", value)) {
        parsed.coarsen_type = static_cast<int>(value);
        saw_field = true;
    }
    if (ExtractJsonNumber(line, "interp_type", value)) {
        parsed.interp_type = static_cast<int>(value);
        saw_field = true;
    }
    if (ExtractJsonNumber(line, "max_row_sum", value)) {
        parsed.max_row_sum = static_cast<amrex::Real>(value);
        saw_field = true;
    }
    if (ExtractJsonNumber(line, "relax_type", value)) {
        parsed.relax_type = static_cast<int>(value);
        saw_field = true;
    }
    if (ExtractJsonNumber(line, "num_sweeps", value)) {
        parsed.num_sweeps = static_cast<int>(value);
        saw_field = true;
    }
    if (ExtractJsonNumber(line, "cycle_type", value)) {
        parsed.cycle_type = static_cast<int>(value);
        saw_field = true;
    }
    if (ExtractJsonNumber(line, "max_levels", value)) {
        parsed.max_levels = static_cast<int>(value);
        saw_field = true;
    }
    if (ExtractJsonNumber(line, "trunc_factor", value)) {
        parsed.trunc_factor = static_cast<amrex::Real>(value);
        saw_field = true;
    }
    if (ExtractJsonNumber(line, "P_max_elmts", value) ||
        ExtractJsonNumber(line, "pmax_elmts", value))
    {
        parsed.pmax_elmts = static_cast<int>(value);
        saw_field = true;
    }
    if (ExtractJsonNumber(line, "agg_num_levels", value)) {
        parsed.agg_num_levels = static_cast<int>(value);
        saw_field = true;
    }
    if (ExtractJsonNumber(line, "agg_interp_type", value)) {
        parsed.agg_interp_type = static_cast<int>(value);
        saw_field = true;
    }
    if (ExtractJsonNumber(line, "agg_tr", value) ||
        ExtractJsonNumber(line, "agg_trunc_factor", value))
    {
        parsed.agg_trunc_factor = static_cast<amrex::Real>(value);
        saw_field = true;
    }
    if (ExtractJsonNumber(line, "agg_Pmx", value) ||
        ExtractJsonNumber(line, "agg_pmax_elmts", value))
    {
        parsed.agg_pmax_elmts = static_cast<int>(value);
        saw_field = true;
    }
    if (ExtractJsonNumber(line, "relax_wt", value)) {
        parsed.relax_wt = static_cast<amrex::Real>(value);
        saw_field = true;
    }
    if (ExtractJsonNumber(line, "relax_order", value)) {
        parsed.relax_order = static_cast<int>(value);
        saw_field = true;
    }
    if (ExtractJsonNumber(line, "max_coarse_size", value)) {
        parsed.max_coarse_size = static_cast<int>(value);
        saw_field = true;
    }

    if (saw_field) {
        action = parsed;
    }
    return saw_field;
}

inline void BroadcastAction (Action& action)
{
    const int root = amrex::ParallelDescriptor::IOProcessorNumber();
    amrex::ParallelDescriptor::Bcast(&action.step, 1, root);
    amrex::ParallelDescriptor::Bcast(&action.nlscoef, 1, root);
    amrex::ParallelDescriptor::Bcast(&action.max_nonlinear_iters, 1, root);
    amrex::ParallelDescriptor::Bcast(&action.eps_lin, 1, root);
    amrex::ParallelDescriptor::Bcast(&action.lsetup_frequency, 1, root);
    amrex::ParallelDescriptor::Bcast(&action.jac_eval_frequency, 1, root);
    amrex::ParallelDescriptor::Bcast(&action.force_lsetup, 1, root);
    amrex::ParallelDescriptor::Bcast(&action.force_jac_eval, 1, root);
}

inline void BroadcastHypreAction (HypreAction& action)
{
    const int root = amrex::ParallelDescriptor::IOProcessorNumber();
    amrex::ParallelDescriptor::Bcast(&action.setup, 1, root);
    amrex::ParallelDescriptor::Bcast(&action.strong_threshold, 1, root);
    amrex::ParallelDescriptor::Bcast(&action.coarsen_type, 1, root);
    amrex::ParallelDescriptor::Bcast(&action.interp_type, 1, root);
    amrex::ParallelDescriptor::Bcast(&action.max_row_sum, 1, root);
    amrex::ParallelDescriptor::Bcast(&action.relax_type, 1, root);
    amrex::ParallelDescriptor::Bcast(&action.num_sweeps, 1, root);
    amrex::ParallelDescriptor::Bcast(&action.cycle_type, 1, root);
    amrex::ParallelDescriptor::Bcast(&action.max_levels, 1, root);
    amrex::ParallelDescriptor::Bcast(&action.trunc_factor, 1, root);
    amrex::ParallelDescriptor::Bcast(&action.pmax_elmts, 1, root);
    amrex::ParallelDescriptor::Bcast(&action.agg_num_levels, 1, root);
    amrex::ParallelDescriptor::Bcast(&action.agg_interp_type, 1, root);
    amrex::ParallelDescriptor::Bcast(&action.agg_trunc_factor, 1, root);
    amrex::ParallelDescriptor::Bcast(&action.agg_pmax_elmts, 1, root);
    amrex::ParallelDescriptor::Bcast(&action.relax_wt, 1, root);
    amrex::ParallelDescriptor::Bcast(&action.relax_order, 1, root);
    amrex::ParallelDescriptor::Bcast(&action.max_coarse_size, 1, root);
}

inline void ReadJsonActionFromStdin (int step, Action& action)
{
    int ok = 1;
    if (amrex::ParallelDescriptor::IOProcessor()) {
        std::string line;
        if (!std::getline(std::cin, line) || !ParseJsonAction(line, step, action)) {
            ok = 0;
        }
    }

    const int root = amrex::ParallelDescriptor::IOProcessorNumber();
    amrex::ParallelDescriptor::Bcast(&ok, 1, root);
    BroadcastAction(action);
    if (ok == 0) {
        amrex::Abort("Expected JSON action on stdin for rl_control.mode=stdin_json");
    }
}

inline void ReadJsonHypreActionFromStdin (int setup, HypreAction& action)
{
    int ok = 1;
    if (amrex::ParallelDescriptor::IOProcessor()) {
        std::string line;
        if (!std::getline(std::cin, line) || !ParseJsonHypreAction(line, setup, action)) {
            ok = 0;
        }
    }

    const int root = amrex::ParallelDescriptor::IOProcessorNumber();
    amrex::ParallelDescriptor::Bcast(&ok, 1, root);
    BroadcastHypreAction(action);
    if (ok == 0) {
        amrex::Abort("Expected JSON HYPRE action on stdin for hypre_control.mode=stdin_json");
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
    delta.precond_setup_calls = after.precond_setup_calls - before.precond_setup_calls;
    delta.precond_setup_wall_seconds =
        after.precond_setup_wall_seconds - before.precond_setup_wall_seconds;
    delta.solve_calls = after.solve_calls - before.solve_calls;
    delta.iterations = after.iterations - before.iterations;
    delta.wall_seconds = after.wall_seconds - before.wall_seconds;
    delta.hypre_setup_calls = after.hypre_setup_calls - before.hypre_setup_calls;
    delta.hypre_solve_calls = after.hypre_solve_calls - before.hypre_solve_calls;
    delta.hypre_setup_wall_seconds =
        after.hypre_setup_wall_seconds - before.hypre_setup_wall_seconds;
    delta.hypre_solve_wall_seconds =
        after.hypre_solve_wall_seconds - before.hypre_solve_wall_seconds;
    delta.last_precond_setup_wall_seconds = after.last_precond_setup_wall_seconds;
    delta.last_iterations = after.last_iterations;
    delta.last_residual = after.last_residual;
    delta.last_hypre_setup_wall_seconds = after.last_hypre_setup_wall_seconds;
    delta.last_hypre_solve_wall_seconds = after.last_hypre_solve_wall_seconds;
    delta.last_hypre_cum_nnz_ap = after.last_hypre_cum_nnz_ap;
    delta.last_hypre_hierarchy_levels = after.last_hypre_hierarchy_levels;
    delta.last_hypre_finest_rows = after.last_hypre_finest_rows;
    delta.last_hypre_coarsest_rows = after.last_hypre_coarsest_rows;
    return delta;
}

inline void RecordMlmgPrecondSetup (MlmgTelemetry& telemetry, amrex::Real wall_seconds)
{
    telemetry.precond_setup_calls += 1;
    telemetry.precond_setup_wall_seconds += wall_seconds;
    telemetry.last_precond_setup_wall_seconds = wall_seconds;
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

inline ActiveDiffusion ActiveDiffusionAt (const RunMetadata& metadata, amrex::Real time)
{
    ActiveDiffusion active;
    active.x = metadata.diffCoeffx;
    active.y = metadata.diffCoeffy;

    const auto count = metadata.diffusion_times.size();
    if (count == 0 || metadata.diffusion_x_multipliers.size() != count ||
        metadata.diffusion_y_multipliers.size() != count)
    {
        return active;
    }

    std::size_t segment = 0;
    for (std::size_t i = 0; i < count; ++i) {
        if (time >= metadata.diffusion_times[i]) {
            segment = i;
        } else {
            break;
        }
    }

    active.segment = static_cast<int>(segment);
    active.x_multiplier = metadata.diffusion_x_multipliers[segment];
    active.y_multiplier = metadata.diffusion_y_multipliers[segment];
    if (metadata.diffusion_interpolation == "linear" && segment + 1 < count) {
        const amrex::Real span = metadata.diffusion_times[segment + 1] -
                                 metadata.diffusion_times[segment];
        const amrex::Real theta = span > amrex::Real(0.0)
            ? std::clamp((time - metadata.diffusion_times[segment]) / span,
                         amrex::Real(0.0), amrex::Real(1.0))
            : amrex::Real(0.0);
        active.x_multiplier =
            (amrex::Real(1.0) - theta) * metadata.diffusion_x_multipliers[segment] +
            theta * metadata.diffusion_x_multipliers[segment + 1];
        active.y_multiplier =
            (amrex::Real(1.0) - theta) * metadata.diffusion_y_multipliers[segment] +
            theta * metadata.diffusion_y_multipliers[segment + 1];
    }
    active.x = metadata.diffCoeffx * active.x_multiplier;
    active.y = metadata.diffCoeffy * active.y_multiplier;
    return active;
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
    os << "{\"precond_setup_calls\":" << stats.precond_setup_calls
       << ",\"precond_setup_wall_seconds\":" << stats.precond_setup_wall_seconds
       << ",\"solve_calls\":" << stats.solve_calls
       << ",\"iterations\":" << stats.iterations
       << ",\"wall_seconds\":" << stats.wall_seconds
       << ",\"hypre_setup_calls\":" << stats.hypre_setup_calls
       << ",\"hypre_solve_calls\":" << stats.hypre_solve_calls
       << ",\"hypre_setup_wall_seconds\":" << stats.hypre_setup_wall_seconds
       << ",\"hypre_solve_wall_seconds\":" << stats.hypre_solve_wall_seconds
       << ",\"last_precond_setup_wall_seconds\":" << stats.last_precond_setup_wall_seconds
       << ",\"last_iterations\":" << stats.last_iterations
       << ",\"last_residual\":" << stats.last_residual
       << ",\"last_hypre_setup_wall_seconds\":" << stats.last_hypre_setup_wall_seconds
       << ",\"last_hypre_solve_wall_seconds\":" << stats.last_hypre_solve_wall_seconds
       << ",\"last_hypre_cum_nnz_ap\":" << stats.last_hypre_cum_nnz_ap
       << ",\"last_hypre_hierarchy_levels\":" << stats.last_hypre_hierarchy_levels
       << ",\"last_hypre_finest_rows\":" << stats.last_hypre_finest_rows
       << ",\"last_hypre_coarsest_rows\":" << stats.last_hypre_coarsest_rows
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

inline void WriteRealArray (std::ostream& os, const std::vector<amrex::Real>& values)
{
    os << "[";
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            os << ",";
        }
        os << values[i];
    }
    os << "]";
}

inline void WriteAction (std::ostream& os, const Action& action)
{
    os << "{\"nlscoef\":" << action.nlscoef
       << ",\"max_nonlinear_iters\":" << action.max_nonlinear_iters
       << ",\"eps_lin\":" << action.eps_lin
       << ",\"lsetup_frequency\":" << action.lsetup_frequency
       << ",\"jac_eval_frequency\":" << action.jac_eval_frequency
       << ",\"force_lsetup\":" << action.force_lsetup
       << ",\"force_jac_eval\":" << action.force_jac_eval
       << "}";
}

inline void WriteHypreAction (std::ostream& os, const HypreAction& action)
{
    os << "{\"strong_threshold\":" << action.strong_threshold
       << ",\"coarsen_type\":" << action.coarsen_type
       << ",\"interp_type\":" << action.interp_type
       << ",\"max_row_sum\":" << action.max_row_sum
       << ",\"relax_type\":" << action.relax_type
       << ",\"num_sweeps\":" << action.num_sweeps
       << ",\"cycle_type\":" << action.cycle_type
       << ",\"max_levels\":" << action.max_levels
       << ",\"trunc_factor\":" << action.trunc_factor
       << ",\"P_max_elmts\":" << action.pmax_elmts
       << ",\"agg_num_levels\":" << action.agg_num_levels
       << ",\"agg_interp_type\":" << action.agg_interp_type
       << ",\"agg_tr\":" << action.agg_trunc_factor
       << ",\"agg_Pmx\":" << action.agg_pmax_elmts
       << ",\"relax_wt\":" << action.relax_wt
       << ",\"relax_order\":" << action.relax_order
       << ",\"max_coarse_size\":" << action.max_coarse_size
       << "}";
}

inline void WriteHypreSetupContext (std::ostream& os, const HypreSetupContext& context)
{
    os << "{\"setup\":" << context.setup
       << ",\"time\":" << context.time
       << ",\"gamma\":" << context.gamma
       << ",\"dt\":" << context.dt
       << ",\"n_cell\":" << context.n_cell
       << ",\"max_grid_size\":" << context.max_grid_size
       << ",\"dimension\":" << context.dimension
       << ",\"dx\":" << context.dx
       << ",\"advCoeffx\":" << context.advCoeffx
       << ",\"advCoeffy\":" << context.advCoeffy
       << ",\"active_diffCoeffx\":" << context.active_diffCoeffx
       << ",\"active_diffCoeffy\":" << context.active_diffCoeffy
       << ",\"cfl_adv_x\":" << context.cfl_adv_x
       << ",\"cfl_adv_y\":" << context.cfl_adv_y
       << ",\"cfl_diff_x\":" << context.cfl_diff_x
       << ",\"cfl_diff_y\":" << context.cfl_diff_y
       << ",\"tol\":" << context.tol
       << ",\"max_iter\":" << context.max_iter
       << "}";
}

inline void WriteHypreSetupRequest (std::ostream& output, const HypreSetupContext& context,
                                    const HypreAction& current_action)
{
    output << "{\"event\":\"hypre_setup_request\""
           << ",\"setup\":" << context.setup
           << ",\"hypre_context\":";
    WriteHypreSetupContext(output, context);
    output << ",\"current_hypre_action\":";
    WriteHypreAction(output, current_action);
    output << "}\n";
}

inline void WriteHypreSetup (std::ostream& output, const HypreSetupContext& context,
                             const HypreAction& action)
{
    output << "{\"event\":\"hypre_setup\""
           << ",\"setup\":" << context.setup
           << ",\"hypre_context\":";
    WriteHypreSetupContext(output, context);
    output << ",\"hypre_action\":";
    WriteHypreAction(output, action);
    output << "}\n";
}

inline void WriteRunStart (std::ostream& output, const amrex::MultiFab& phi,
                           const RunMetadata& metadata, const Action& initial_action)
{
    const PhiSummary initial_phi = SummarizePhi(phi);

    output << "{\"event\":\"run_start\""
           << ",\"n_cell\":" << metadata.n_cell
           << ",\"max_grid_size\":" << metadata.max_grid_size
           << ",\"max_agent_steps\":" << metadata.max_agent_steps
           << ",\"final_time\":" << metadata.final_time
           << ",\"dt_nominal\":" << metadata.dt_nominal
           << ",\"adapt_dt\":" << (metadata.adapt_dt ? "true" : "false")
           << ",\"advCoeffx\":" << metadata.advCoeffx
           << ",\"advCoeffy\":" << metadata.advCoeffy
           << ",\"diffCoeffx\":" << metadata.diffCoeffx
           << ",\"diffCoeffy\":" << metadata.diffCoeffy
           << ",\"diffusion_times\":";
    WriteRealArray(output, metadata.diffusion_times);
    output << ",\"diffusion_x_multipliers\":";
    WriteRealArray(output, metadata.diffusion_x_multipliers);
    output << ",\"diffusion_y_multipliers\":";
    WriteRealArray(output, metadata.diffusion_y_multipliers);
    output
           << ",\"diffusion_interpolation\":\""
           << JsonEscape(metadata.diffusion_interpolation) << "\""
           << ",\"diffusion_reinitialize_at_jumps\":"
           << (metadata.diffusion_reinitialize_at_jumps ? "true" : "false")
           << ",\"initial_nlscoef\":" << initial_action.nlscoef
           << ",\"initial_max_nonlinear_iters\":"
           << initial_action.max_nonlinear_iters
           << ",\"initial_eps_lin\":" << initial_action.eps_lin
           << ",\"initial_lsetup_frequency\":"
           << initial_action.lsetup_frequency
           << ",\"initial_jac_eval_frequency\":"
           << initial_action.jac_eval_frequency
           << ",\"initial_force_lsetup\":" << initial_action.force_lsetup
           << ",\"initial_force_jac_eval\":" << initial_action.force_jac_eval
           << ",\"schedule\":\"" << JsonEscape(metadata.schedule_path) << "\""
           << ",\"control_mode\":\"" << JsonEscape(metadata.control_mode) << "\""
           << ",\"phi_initial\":";
    WritePhiSummary(output, initial_phi);
    output << "}\n";
}

inline void RecordRunStart (std::ostream* output, const amrex::MultiFab& phi,
                            const RunMetadata& metadata, const Action& initial_action,
                            bool enabled)
{
    if (!enabled || output == nullptr) {
        return;
    }
    WriteRunStart(*output, phi, metadata, initial_action);
}

inline void WriteControlRequest (std::ostream& output, int control_step,
                                 amrex::Real time_start, amrex::Real time_limit,
                                 amrex::Real dt_proposed,
                                 const Action& current_action,
                                 const StepSnapshot& snapshot,
                                 const RunMetadata& metadata)
{
    const auto active = ActiveDiffusionAt(metadata, time_start);
    output << "{\"event\":\"control_request\""
           << ",\"control_step\":" << control_step
           << ",\"time_start\":" << time_start
           << ",\"time_limit\":" << time_limit
           << ",\"dt_proposed\":" << dt_proposed
           << ",\"final_time\":" << metadata.final_time
           << ",\"n_cell\":" << metadata.n_cell
           << ",\"max_grid_size\":" << metadata.max_grid_size
           << ",\"max_agent_steps\":" << metadata.max_agent_steps
           << ",\"adapt_dt\":" << (metadata.adapt_dt ? "true" : "false")
           << ",\"advCoeffx\":" << metadata.advCoeffx
           << ",\"advCoeffy\":" << metadata.advCoeffy
           << ",\"diffCoeffx\":" << metadata.diffCoeffx
           << ",\"diffCoeffy\":" << metadata.diffCoeffy
           << ",\"active_diffusion_segment\":" << active.segment
           << ",\"active_diffusion_x_multiplier\":" << active.x_multiplier
           << ",\"active_diffusion_y_multiplier\":" << active.y_multiplier
           << ",\"active_diffCoeffx\":" << active.x
           << ",\"active_diffCoeffy\":" << active.y
           << ",\"current_action\":";
    WriteAction(output, current_action);
    output << ",\"phi\":";
    WritePhiSummary(output, snapshot.phi);
    output << ",\"sundials\":";
    WriteSundialsStats(output, snapshot.sundials);
    output << ",\"mlmg\":";
    WriteMlmgTelemetry(output, snapshot.mlmg);
    output << "}\n";
}

inline void WriteTransition (std::ostream& output, int control_step,
                             amrex::Real time_start, amrex::Real time_limit,
                             amrex::Real time_after, amrex::Real dt_actual,
                             amrex::Real dt_proposed,
                             const Action& action,
                             amrex::Real step_wall_seconds, const StepSnapshot& before,
                             const StepSnapshot& after, const RunMetadata& metadata)
{
    const auto sundials_delta = after.sundials - before.sundials;
    const auto mlmg_delta = after.mlmg - before.mlmg;
    const bool step_valid = StepIsValid(after);
    const auto active = ActiveDiffusionAt(metadata, time_start);
    const auto active_after = ActiveDiffusionAt(metadata, time_after);
    const bool crossed_diffusion_jump =
        metadata.diffusion_interpolation == "constant" &&
        active.segment != active_after.segment;

    output << "{\"event\":\"transition\""
           << ",\"control_step\":" << control_step
           << ",\"time_start\":" << time_start
           << ",\"time_limit\":" << time_limit
           << ",\"time_after\":" << time_after
           << ",\"dt_actual\":" << dt_actual
           << ",\"dt_proposed\":" << dt_proposed
           << ",\"final_time\":" << metadata.final_time
           << ",\"active_diffusion_segment\":" << active.segment
           << ",\"active_diffusion_x_multiplier\":" << active.x_multiplier
           << ",\"active_diffusion_y_multiplier\":" << active.y_multiplier
           << ",\"active_diffCoeffx\":" << active.x
           << ",\"active_diffCoeffy\":" << active.y
           << ",\"crossed_diffusion_jump\":"
           << (crossed_diffusion_jump ? "true" : "false")
           << ",\"action\":";
    WriteAction(output, action);
    output
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

inline bool RecordTransition (std::ostream* output, int control_step,
                              amrex::Real time_start, amrex::Real time_limit,
                              amrex::Real time_after, amrex::Real dt_actual,
                              amrex::Real dt_proposed,
                              const Action& action,
                              amrex::Real step_wall_seconds, const StepSnapshot& before,
                              const StepSnapshot& after, const RunMetadata& metadata)
{
    if (output != nullptr) {
        WriteTransition(*output, control_step, time_start, time_limit, time_after,
                        dt_actual, dt_proposed, action, step_wall_seconds, before, after,
                        metadata);
    }
    return StepIsValid(after);
}

inline bool WriteRunEnd (std::ostream& output, const amrex::MultiFab& phi, int completed_steps,
                         bool episode_valid, amrex::Real time_final,
                         amrex::Real evolution_wall_seconds, amrex::Real analytic_l1_error,
                         const MlmgTelemetry& mlmg, const RunMetadata& metadata)
{
    const PhiSummary final_phi = SummarizePhi(phi);
    const bool valid = episode_valid && final_phi.finite;

    output << "{\"event\":\"run_end\""
           << ",\"completed_steps\":" << completed_steps
           << ",\"max_agent_steps\":" << metadata.max_agent_steps
           << ",\"time_final\":" << time_final
           << ",\"final_time\":" << metadata.final_time
           << ",\"valid\":" << (valid ? "true" : "false")
           << ",\"total_evolution_time\":" << evolution_wall_seconds
           << ",\"analytic_l1_error\":";
    if (std::isfinite(analytic_l1_error)) {
        output << analytic_l1_error;
    } else {
        output << "null";
    }
    output << ",\"mlmg_total\":";
    WriteMlmgTelemetry(output, mlmg);
    output << ",\"phi_final\":";
    WritePhiSummary(output, final_phi);
    output << "}\n";

    return valid;
}

inline bool RecordRunEnd (std::ostream* output, const amrex::MultiFab& phi, int completed_steps,
                          bool episode_valid, amrex::Real time_final,
                          amrex::Real evolution_wall_seconds, amrex::Real analytic_l1_error,
                          const MlmgTelemetry& mlmg, const RunMetadata& metadata)
{
    if (output != nullptr) {
        return WriteRunEnd(*output, phi, completed_steps, episode_valid, time_final,
                           evolution_wall_seconds, analytic_l1_error, mlmg, metadata);
    }

    const PhiSummary final_phi = SummarizePhi(phi);
    const bool valid = episode_valid && final_phi.finite;
    return valid;
}

} // namespace AdvDiffRl

#endif
