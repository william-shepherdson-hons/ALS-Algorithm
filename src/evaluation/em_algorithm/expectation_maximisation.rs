use crate::evaluation::em_algorithm::iteration::Iteration;
use crate::evaluation::em_algorithm::em_result::EmResult;
use crate::models::models::Models;
use crate::evaluation::em_algorithm::formatted_record::FormattedRecord;
use std::{collections::HashMap, error::Error};
use csv::{Writer, ReaderBuilder};

struct UserSkillSequence {
    observations: Vec<bool>,
}

struct ExpectedCounts {
    sum_initial_mastery: f64,
    sum_learned: f64,
    sum_opportunities_unknown: f64,
    sum_correct_while_known: f64,
    sum_known: f64,
    sum_correct_while_unknown: f64,
    sum_unknown: f64,
    n_sequences: usize,
}

impl ExpectedCounts {
    fn new() -> Self {
        Self {
            sum_initial_mastery: 0.0,
            sum_learned: 0.0,
            sum_opportunities_unknown: 0.0,
            sum_correct_while_known: 0.0,
            sum_known: 0.0,
            sum_correct_while_unknown: 0.0,
            sum_unknown: 0.0,
            n_sequences: 0,
        }
    }
}

/// Compute emission probability P(observation | state)
fn emission_probability(state_known: bool, observed_correct: bool, params: &EmResult) -> f64 {
    match (state_known, observed_correct) {
        (true, true) => 1.0 - params.slip,      // Known and correct
        (true, false) => params.slip,            // Known but incorrect (slip)
        (false, true) => params.guess,           // Unknown but correct (guess)
        (false, false) => 1.0 - params.guess,    // Unknown and incorrect
    }
}

/// Get transition probabilities based on model type
fn get_transition_probs(model: &Models, params: &EmResult) -> (f64, f64, f64, f64) {
    match model {
        Models::HiddenMarkovModel => {
            // Symmetric transitions
            let p_kk = params.transition;      // known -> known
            let p_ku = 1.0 - params.transition; // known -> unknown
            let p_uk = params.transition;       // unknown -> known (learn)
            let p_uu = 1.0 - params.transition; // unknown -> unknown
            (p_kk, p_ku, p_uk, p_uu)
        }
        Models::KnowledgeTracingModel => {
            // One-way transitions (can't unlearn)
            let p_kk = 1.0;                     // known -> known (always)
            let p_ku = 0.0;                     // known -> unknown (never)
            let p_uk = params.transition;       // unknown -> known (learn)
            let p_uu = 1.0 - params.transition; // unknown -> unknown
            (p_kk, p_ku, p_uk, p_uu)
        }
    }
}

/// Forward pass: compute alpha values (filtered probabilities)
/// Returns (alpha_known, alpha_unknown) for each time step
async fn forward_pass(
    observations: &[bool],
    params: &EmResult,
    model: &Models,
) -> (Vec<f64>, Vec<f64>) {
    let n = observations.len();
    let mut alpha_known = Vec::with_capacity(n + 1);
    let mut alpha_unknown = Vec::with_capacity(n + 1);

    // Initialize with prior
    alpha_known.push(params.initial);
    alpha_unknown.push(1.0 - params.initial);

    let (p_kk, p_ku, p_uk, p_uu) = get_transition_probs(model, params);

    for &obs in observations {
        let prev_k = *alpha_known.last().unwrap();
        let prev_u = *alpha_unknown.last().unwrap();

        // Emission probabilities
        let e_known = emission_probability(true, obs, params);
        let e_unknown = emission_probability(false, obs, params);

        // Forward equations: alpha(t) = P(obs(t) | state) * sum_s P(state | s) * alpha(s, t-1)
        let new_k = e_known * (prev_k * p_kk + prev_u * p_uk);
        let new_u = e_unknown * (prev_k * p_ku + prev_u * p_uu);

        // Normalize to prevent numerical underflow
        let norm = new_k + new_u;
        if norm > 0.0 {
            alpha_known.push(new_k / norm);
            alpha_unknown.push(new_u / norm);
        } else {
            // Fallback if both are zero (shouldn't happen with proper parameters)
            alpha_known.push(prev_k);
            alpha_unknown.push(prev_u);
        }
    }

    (alpha_known, alpha_unknown)
}

/// Backward pass: compute beta values (backward messages)
/// Returns (beta_known, beta_unknown) for each time step
async fn backward_pass(
    observations: &[bool],
    params: &EmResult,
    model: &Models,
) -> (Vec<f64>, Vec<f64>) {
    let n = observations.len();
    let mut beta_known = vec![0.0; n + 1];
    let mut beta_unknown = vec![0.0; n + 1];

    // Initialize: beta(T) = 1 for all states
    beta_known[n] = 1.0;
    beta_unknown[n] = 1.0;

    let (p_kk, p_ku, p_uk, p_uu) = get_transition_probs(model, params);

    // Backward recursion
    for t in (0..n).rev() {
        let obs = observations[t];

        let e_known = emission_probability(true, obs, params);
        let e_unknown = emission_probability(false, obs, params);

        // Backward equations: beta(t-1) = sum_s P(state(t)=s | state(t-1)) * P(obs(t) | s) * beta(s, t)
        beta_known[t] = p_kk * e_known * beta_known[t + 1] + p_ku * e_unknown * beta_unknown[t + 1];
        beta_unknown[t] = p_uk * e_known * beta_known[t + 1] + p_uu * e_unknown * beta_unknown[t + 1];

        // Normalize to prevent overflow/underflow
        let norm = beta_known[t] + beta_unknown[t];
        if norm > 0.0 {
            beta_known[t] /= norm;
            beta_unknown[t] /= norm;
        }
    }

    (beta_known, beta_unknown)
}

/// Compute smoothed probabilities: P(state | all observations)
fn compute_gamma(
    alpha_known: &[f64],
    alpha_unknown: &[f64],
    beta_known: &[f64],
    beta_unknown: &[f64],
) -> Vec<f64> {
    let n = alpha_known.len();
    let mut gamma_known = Vec::with_capacity(n);

    for t in 0..n {
        let gamma_k = alpha_known[t] * beta_known[t];
        let gamma_u = alpha_unknown[t] * beta_unknown[t];
        let total = gamma_k + gamma_u;

        let prob_known = if total > 0.0 {
            gamma_k / total
        } else {
            alpha_known[t] // Fallback to filtered estimate
        };

        gamma_known.push(prob_known.min(1.0).max(0.0));
    }

    gamma_known
}

/// Compute transition expectations: P(state(t-1)=i, state(t)=j | all observations)
/// Returns xi[t] = P(unknown at t-1, known at t | all obs) for each transition
async fn compute_xi(
    observations: &[bool],
    alpha_known: &[f64],
    alpha_unknown: &[f64],
    beta_known: &[f64],
    beta_unknown: &[f64],
    params: &EmResult,
    model: &Models,
) -> Vec<f64> {
    let n = observations.len();
    let mut xi_uk = Vec::with_capacity(n); // unknown -> known transitions

    let (p_kk, p_ku, p_uk, p_uu) = get_transition_probs(model, params);

    for t in 0..n {
        let obs = observations[t];

        let e_known = emission_probability(true, obs, params);
        let e_unknown = emission_probability(false, obs, params);

        // Joint probability of each transition and observation at t
        let j_kk = alpha_known[t] * p_kk * e_known * beta_known[t + 1];
        let j_ku = alpha_known[t] * p_ku * e_unknown * beta_unknown[t + 1];
        let j_uk = alpha_unknown[t] * p_uk * e_known * beta_known[t + 1];
        let j_uu = alpha_unknown[t] * p_uu * e_unknown * beta_unknown[t + 1];

        let total = j_kk + j_ku + j_uk + j_uu;

        let prob_uk = if total > 0.0 {
            j_uk / total
        } else {
            0.0
        };

        xi_uk.push(prob_uk.min(1.0).max(0.0));
    }

    xi_uk
}

/// Accumulate expected counts from one sequence (E-step)
async fn accumulate_counts(
    observations: &[bool],
    params: &EmResult,
    model: &Models,
    counts: &mut ExpectedCounts,
) {
    if observations.is_empty() {
        return;
    }

    // Forward-backward algorithm
    let (alpha_known, alpha_unknown) = forward_pass(observations, params, model).await;
    let (beta_known, beta_unknown) = backward_pass(observations, params, model).await;

    // Compute smoothed state probabilities
    let gamma_known = compute_gamma(&alpha_known, &alpha_unknown, &beta_known, &beta_unknown);

    // Compute transition expectations
    let xi_uk = compute_xi(
        observations,
        &alpha_known,
        &alpha_unknown,
        &beta_known,
        &beta_unknown,
        params,
        model,
    )
    .await;

    // Accumulate initial state counts
    counts.sum_initial_mastery += gamma_known[0];
    counts.n_sequences += 1;

    // Accumulate transition and emission counts
    for t in 0..observations.len() {
        let prob_known = gamma_known[t];
        let prob_unknown = 1.0 - prob_known;

        // Transition counts (learning)
        counts.sum_learned += xi_uk[t];
        counts.sum_opportunities_unknown += prob_unknown;

        // Emission counts
        if observations[t] {
            counts.sum_correct_while_known += prob_known;
            counts.sum_correct_while_unknown += prob_unknown;
        }

        counts.sum_known += prob_known;
        counts.sum_unknown += prob_unknown;
    }
}

/// M-step: update parameters based on expected counts
fn m_step(counts: &ExpectedCounts) -> EmResult {
    let initial = if counts.n_sequences > 0 {
        (counts.sum_initial_mastery / counts.n_sequences as f64)
            .min(0.99)
            .max(0.01)
    } else {
        0.5
    };

    let transition = if counts.sum_opportunities_unknown > 0.0 {
        (counts.sum_learned / counts.sum_opportunities_unknown)
            .min(0.99)
            .max(0.01)
    } else {
        0.1
    };

    let slip = if counts.sum_known > 0.0 {
        (1.0 - counts.sum_correct_while_known / counts.sum_known)
            .min(0.99)
            .max(0.01)
    } else {
        0.1
    };

    let guess = if counts.sum_unknown > 0.0 {
        (counts.sum_correct_while_unknown / counts.sum_unknown)
            .min(0.99)
            .max(0.01)
    } else {
        0.2
    };

    // Ensure guess + slip < 1 (identifiability constraint)
    let sum = guess + slip;
    let (guess, slip) = if sum >= 1.0 {
        let scale = 0.98 / sum;
        (guess * scale, slip * scale)
    } else {
        (guess, slip)
    };

    EmResult {
        initial,
        transition,
        slip,
        guess,
    }
}

/// Main EM algorithm
pub async fn expectation_maximisation(
    model: Models,
    initial: EmResult,
    path: &str,
    output: &str,
) -> Result<EmResult, Box<dyn Error>> {
    // Load and parse data
    let mut reader = ReaderBuilder::new()
        .trim(csv::Trim::All)
        .from_path(path)?;
    let mut writer = Writer::from_path(output)?;
    let mut records: Vec<FormattedRecord> = reader
        .deserialize()
        .collect::<Result<_, _>>()?;

    // Sort by user, skill, and time
    records.sort_by_key(|r| (r.user_id, r.skill_id, r.times_applied));

    // Group into sequences by (user_id, skill_id)
    let mut sequences: HashMap<(u32, u32), UserSkillSequence> = HashMap::new();
    for record in records {
        let key = (record.user_id, record.skill_id);
        let sequence = sequences
            .entry(key)
            .or_insert_with(|| UserSkillSequence {
                observations: Vec::new(),
            });
        sequence.observations.push(record.correct == 1);
    }

    const MAX_ITERATIONS: usize = 1000;
    const TOLERANCE: f64 = 1e-4;
    let mut params = initial;

    println!("Starting EM with {} sequences", sequences.len());
    println!(
        "Initial params: L0={:.4}, T={:.4}, S={:.4}, G={:.4}",
        params.initial, params.transition, params.slip, params.guess
    );

    // EM iterations
    for iteration in 0..MAX_ITERATIONS {
        let mut counts = ExpectedCounts::new();

        // E-step: compute expected counts from all sequences
        for (_key, sequence) in &sequences {
            accumulate_counts(&sequence.observations, &params, &model, &mut counts).await;
        }

        // M-step: update parameters
        let new_params = m_step(&counts);

        // Check convergence
        let diff = (new_params.initial - params.initial).abs()
            + (new_params.transition - params.transition).abs()
            + (new_params.slip - params.slip).abs()
            + (new_params.guess - params.guess).abs();

        println!(
            "Iteration {}: L0={:.4}, T={:.4}, S={:.4}, G={:.4}, diff={:.6}",
            iteration + 1,
            new_params.initial,
            new_params.transition,
            new_params.slip,
            new_params.guess,
            diff
        );

        // Write iteration results
        let iteration_output = Iteration {
            iteration,
            initial: new_params.initial,
            transition: new_params.transition,
            slip: new_params.slip,
            guess: new_params.guess,
            diff,
        };
        writer.serialize(&iteration_output)?;
        writer.flush()?;

        params = new_params;

        if diff < TOLERANCE {
            println!("Converged after {} iterations", iteration + 1);
            break;
        }
    }

    Ok(params)
}