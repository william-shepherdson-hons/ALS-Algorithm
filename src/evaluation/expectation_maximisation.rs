use crate::models::knowledge_tracing_model;
use crate::{evaluation::em_result::EmResult, models::hidden_markov_model};
use crate::models::models::Models;
use crate::evaluation::formatted_record::FormattedRecord;
use std::{collections::HashMap, error::Error};
use csv::ReaderBuilder;

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

/// Emission probability helper
fn emission_prob(state_known: bool, obs: bool, slip: f64, guess: f64) -> f64 {
    if obs {
        if state_known { 1.0 - slip } else { guess }
    } else {
        if state_known { slip } else { 1.0 - guess }
    }
}

/// Forward pass - computes alpha values (filtering probabilities)
/// Returns P(Known_t | obs_1:t) for t=1..T
fn forward_pass(observations: &[bool], params: &EmResult, model: &Models) -> Vec<f64> {
    let n = observations.len();
    let mut alpha = Vec::with_capacity(n + 1);
    
    // Initial distribution (before any observations)
    alpha.push(params.initial);
    
    for t in 0..n {
        let obs_t = observations[t];
        let alpha_k_prev = alpha[t];
        let alpha_u_prev = 1.0 - alpha_k_prev;
        
        let e_k = emission_prob(true, obs_t, params.slip, params.guess);
        let e_u = emission_prob(false, obs_t, params.slip, params.guess);
        
        let (p_kk, p_uk) = match model {
            Models::HiddenMarkovModel => {
                // For HMM: both states can transition to both states
                // p_kk = probability of staying known
                // p_uk = probability of learning (unknown -> known)
                // Use different rates: high retention, moderate learning
                let retention = 0.95; // High probability of retaining knowledge
                let learning = params.transition;
                (retention, learning)
            }
            Models::KnowledgeTracingModel => {
                // For KT: once known, always known
                (1.0, params.transition)
            }
        };
        
        let p_ku = 1.0 - p_kk;
        let p_uu = 1.0 - p_uk;
        
        // Predict next state
        let pred_k = alpha_k_prev * p_kk + alpha_u_prev * p_uk;
        let pred_u = alpha_k_prev * p_ku + alpha_u_prev * p_uu;
        
        // Update with observation
        let num_k = pred_k * e_k;
        let num_u = pred_u * e_u;
        let normalizer = num_k + num_u;
        
        let alpha_k = if normalizer > 0.0 {
            (num_k / normalizer).max(0.0).min(1.0)
        } else {
            alpha_k_prev
        };
        
        alpha.push(alpha_k);
    }
    
    alpha
}

/// Backward pass - computes beta values
/// Returns (beta_known, beta_unknown) for t=0..T
fn backward_pass(
    observations: &[bool],
    params: &EmResult,
    model: &Models,
) -> (Vec<f64>, Vec<f64>) {
    let n = observations.len();
    let mut beta_k = vec![0.0; n + 1];
    let mut beta_u = vec![0.0; n + 1];
    
    // Initialize at time T (after all observations)
    beta_k[n] = 1.0;
    beta_u[n] = 1.0;
    
    for t in (0..n).rev() {
        let obs_t = observations[t];
        let e_k = emission_prob(true, obs_t, params.slip, params.guess);
        let e_u = emission_prob(false, obs_t, params.slip, params.guess);
        
        let (p_kk, p_uk) = match model {
            Models::HiddenMarkovModel => {
                let retention = 0.95;
                let learning = params.transition;
                (retention, learning)
            }
            Models::KnowledgeTracingModel => {
                (1.0, params.transition)
            }
        };
        
        let p_ku = 1.0 - p_kk;
        let p_uu = 1.0 - p_uk;
        
        beta_k[t] = p_kk * e_k * beta_k[t + 1] + p_ku * e_u * beta_u[t + 1];
        beta_u[t] = p_uk * e_k * beta_k[t + 1] + p_uu * e_u * beta_u[t + 1];
        
        beta_k[t] = beta_k[t].max(1e-100);
        beta_u[t] = beta_u[t].max(1e-100);
    }
    
    (beta_k, beta_u)
}

/// Compute gamma: P(Known_t | all observations)
fn compute_gamma(
    alpha: &[f64],
    beta_k: &[f64],
    beta_u: &[f64],
) -> Vec<f64> {
    let n = alpha.len();
    let mut gamma = Vec::with_capacity(n);
    
    for t in 0..n {
        let alpha_k = alpha[t];
        let alpha_u = 1.0 - alpha_k;
        
        let num_k = alpha_k * beta_k[t];
        let num_u = alpha_u * beta_u[t];
        let denom = num_k + num_u;
        
        let gamma_k = if denom > 0.0 {
            (num_k / denom).max(0.0).min(1.0)
        } else {
            alpha_k
        };
        
        gamma.push(gamma_k);
    }
    
    gamma
}

/// Compute xi: P(Unknown_t, Known_{t+1} | all observations) - the transition probability
fn compute_xi(
    observations: &[bool],
    alpha: &[f64],
    beta_k: &[f64],
    beta_u: &[f64],
    params: &EmResult,
    model: &Models,
) -> Vec<f64> {
    let n = observations.len();
    let mut xi = Vec::with_capacity(n);
    
    for t in 0..n {
        let obs_t = observations[t];
        let alpha_k_t = alpha[t];
        let alpha_u_t = 1.0 - alpha_k_t;
        
        let e_k = emission_prob(true, obs_t, params.slip, params.guess);
        let e_u = emission_prob(false, obs_t, params.slip, params.guess);
        
        let (p_kk, p_uk) = match model {
            Models::HiddenMarkovModel => {
                let retention = 0.95;
                let learning = params.transition;
                (retention, learning)
            }
            Models::KnowledgeTracingModel => {
                (1.0, params.transition)
            }
        };
        
        let p_ku = 1.0 - p_kk;
        let p_uu = 1.0 - p_uk;
        
        // Joint probabilities of all four state transitions
        let j_kk = alpha_k_t * p_kk * e_k * beta_k[t + 1];
        let j_ku = alpha_k_t * p_ku * e_u * beta_u[t + 1];
        let j_uk = alpha_u_t * p_uk * e_k * beta_k[t + 1];
        let j_uu = alpha_u_t * p_uu * e_u * beta_u[t + 1];
        
        let denom = j_kk + j_ku + j_uk + j_uu;
        
        // xi[t] = P(Unknown_t, Known_{t+1} | all obs) = transition from unknown to known
        let xi_t = if denom > 0.0 {
            (j_uk / denom).max(0.0).min(1.0)
        } else {
            0.0
        };
        
        xi.push(xi_t);
    }
    
    xi
}

/// Accumulate expected counts for one sequence
fn accumulate_sequence_counts(
    observations: &[bool],
    alpha: &[f64],
    beta_k: &[f64],
    beta_u: &[f64],
    params: &EmResult,
    model: &Models,
    counts: &mut ExpectedCounts,
) {
    let gamma = compute_gamma(alpha, beta_k, beta_u);
    let xi = compute_xi(observations, alpha, beta_k, beta_u, params, model);
    
    // Initial state: P(Known_0 | all observations)
    counts.sum_initial_mastery += gamma[0];
    counts.n_sequences += 1;
    
    // Accumulate over time steps
    for t in 0..observations.len() {
        // gamma[t] is the state BEFORE observation t
        // gamma[t+1] is the state AFTER observation t
        let gamma_k_before = gamma[t];
        let gamma_u_before = 1.0 - gamma_k_before;
        let gamma_k_after = gamma[t + 1];
        let gamma_u_after = 1.0 - gamma_k_after;
        
        // Transitions: expected number of unknown -> known transitions
        counts.sum_learned += xi[t];
        
        // Opportunities to learn: expected time in unknown state before this observation
        counts.sum_opportunities_unknown += gamma_u_before;
        
        // Emission counts: state AFTER incorporating this observation
        if observations[t] {
            counts.sum_correct_while_known += gamma_k_after;
            counts.sum_correct_while_unknown += gamma_u_after;
        }
        
        counts.sum_known += gamma_k_after;
        counts.sum_unknown += gamma_u_after;
    }
}

/// M-step: update parameters
fn m_step_update(counts: &ExpectedCounts) -> EmResult {
    let initial = if counts.n_sequences > 0 {
        (counts.sum_initial_mastery / counts.n_sequences as f64)
            .min(0.95)
            .max(0.05)
    } else {
        0.5
    };
    
    let transition = if counts.sum_opportunities_unknown > 0.0 {
        (counts.sum_learned / counts.sum_opportunities_unknown)
            .min(0.90)
            .max(0.01)
    } else {
        0.1
    };
    
    // Slip: P(incorrect | known) = 1 - P(correct | known)
    let slip = if counts.sum_known > 0.0 {
        (1.0 - counts.sum_correct_while_known / counts.sum_known)
            .min(0.40)
            .max(0.01)
    } else {
        0.1
    };
    
    // Guess: P(correct | unknown)
    let guess = if counts.sum_unknown > 0.0 {
        (counts.sum_correct_while_unknown / counts.sum_unknown)
            .min(0.40)
            .max(0.01)
    } else {
        0.2
    };
    
    // Ensure slip + guess < 0.9 (identifiability constraint)
    let sum = guess + slip;
    let (guess, slip) = if sum >= 0.80 {
        let scale = 0.75 / sum;
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

/// Compute log-likelihood of data given parameters
fn compute_log_likelihood(
    sequences: &HashMap<(u32, u32), UserSkillSequence>,
    params: &EmResult,
    model: &Models,
) -> f64 {
    let mut total_ll = 0.0;
    
    for (_key, sequence) in sequences {
        let n = sequence.observations.len();
        
        // Use forward algorithm to compute likelihood
        let mut current_known = params.initial;
        
        for (t, &obs) in sequence.observations.iter().enumerate() {
            let e_k = emission_prob(true, obs, params.slip, params.guess);
            let e_u = emission_prob(false, obs, params.slip, params.guess);
            
            let current_unknown = 1.0 - current_known;
            
            // P(obs_t | history) = sum over states of P(state) * P(obs | state)
            let prob_obs = current_known * e_k + current_unknown * e_u;
            
            if prob_obs > 1e-10 {
                total_ll += prob_obs.ln();
            } else {
                total_ll += (-10.0_f64).ln(); // Penalty for very unlikely observations
            }
            
            // Update belief about knowledge state after this observation
            let (p_kk, p_uk) = match model {
                Models::HiddenMarkovModel => {
                    let retention = 0.95;
                    let learning = params.transition;
                    (retention, learning)
                }
                Models::KnowledgeTracingModel => {
                    (1.0, params.transition)
                }
            };
            
            let p_ku = 1.0 - p_kk;
            let p_uu = 1.0 - p_uk;
            
            // Bayes update
            let num_k = current_known * e_k;
            let num_u = current_unknown * e_u;
            let normalizer = num_k + num_u;
            
            let posterior_known = if normalizer > 0.0 {
                num_k / normalizer
            } else {
                current_known
            };
            
            // Transition to next time step
            let posterior_unknown = 1.0 - posterior_known;
            current_known = posterior_known * p_kk + posterior_unknown * p_uk;
            current_known = current_known.max(0.0).min(1.0);
        }
    }
    
    total_ll
}

/// Main EM loop with multiple random initializations
pub async fn expectation_maximisation(
    model: Models,
    initial: EmResult,
    path: &str,
) -> Result<EmResult, Box<dyn Error>> {
    let mut reader = ReaderBuilder::new().trim(csv::Trim::All).from_path(path)?;
    let mut records: Vec<FormattedRecord> = reader.deserialize().collect::<Result<_, _>>()?;
    records.sort_by_key(|r| (r.user_id, r.skill_id, r.times_applied));
    
    let mut sequences: HashMap<(u32, u32), UserSkillSequence> = HashMap::new();
    for record in records {
        let key = (record.user_id, record.skill_id);
        let sequence = sequences.entry(key).or_insert_with(|| UserSkillSequence {
            observations: Vec::new(),
        });
        sequence.observations.push(record.correct == 1);
    }
    
    println!("Starting EM with {} sequences", sequences.len());
    println!("Model: {:?}", model);
    
    // Try multiple random initializations to avoid local minima
    let initializations = vec![
        EmResult { initial: 0.2, transition: 0.15, slip: 0.1, guess: 0.25 }, // Near true params
        EmResult { initial: 0.3, transition: 0.2, slip: 0.15, guess: 0.2 },
        EmResult { initial: 0.15, transition: 0.1, slip: 0.05, guess: 0.3 },
        EmResult { initial: 0.4, transition: 0.25, slip: 0.12, guess: 0.18 },
        EmResult { initial: 0.5, transition: 0.3, slip: 0.2, guess: 0.15 },
        initial.clone(), // User's initial guess
    ];
    
    let mut best_params = initial.clone();
    let mut best_ll = f64::NEG_INFINITY;
    
    for (init_idx, init_params) in initializations.iter().enumerate() {
        println!("\n=== Initialization {} ===", init_idx);
        println!("Start: L0={:.4}, T={:.4}, S={:.4}, G={:.4}",
                 init_params.initial, init_params.transition, 
                 init_params.slip, init_params.guess);
        
        let result = run_em_iteration(&sequences, init_params.clone(), &model)?;
        
        if result.1 > best_ll {
            best_ll = result.1;
            best_params = result.0;
            println!("*** New best log-likelihood: {:.2} ***", best_ll);
        }
    }
    
    println!("\n=== Final Best Parameters ===");
    println!("L0={:.4}, T={:.4}, S={:.4}, G={:.4}", 
             best_params.initial, best_params.transition,
             best_params.slip, best_params.guess);
    println!("Log-likelihood: {:.2}", best_ll);
    
    Ok(best_params)
}

/// Run EM from a single initialization
fn run_em_iteration(
    sequences: &HashMap<(u32, u32), UserSkillSequence>,
    initial: EmResult,
    model: &Models,
) -> Result<(EmResult, f64), Box<dyn Error>> {
    const MAX_ITERATIONS: usize = 200;
    const TOLERANCE: f64 = 1e-4;
    
    let mut params = initial;
    let mut prev_ll = compute_log_likelihood(sequences, &params, model);
    
    println!("Initial LL: {:.2}", prev_ll);
    
    for iteration in 0..MAX_ITERATIONS {
        let mut counts = ExpectedCounts::new();
        
        // E-step
        for (_key, sequence) in sequences {
            let alpha = forward_pass(&sequence.observations, &params, model);
            let (beta_k, beta_u) = backward_pass(&sequence.observations, &params, model);
            
            accumulate_sequence_counts(
                &sequence.observations,
                &alpha,
                &beta_k,
                &beta_u,
                &params,
                model,
                &mut counts,
            );
        }
        
        // M-step
        let new_params = m_step_update(&counts);
        
        // Compute log-likelihood
        let new_ll = compute_log_likelihood(sequences, &new_params, model);
        
        let param_diff = (new_params.initial - params.initial).abs()
            + (new_params.transition - params.transition).abs()
            + (new_params.slip - params.slip).abs()
            + (new_params.guess - params.guess).abs();
        
        let ll_change = new_ll - prev_ll;
        
        if iteration % 10 == 0 || iteration < 10 {
            println!(
                "Iter {}: L0={:.4}, T={:.4}, S={:.4}, G={:.4}, LL={:.2}, ΔLL={:.4}, Δparam={:.6}",
                iteration, new_params.initial, new_params.transition, 
                new_params.slip, new_params.guess, new_ll, ll_change, param_diff
            );
        }
        
        // Check for convergence
        if param_diff < TOLERANCE && ll_change.abs() < 0.1 {
            println!("Converged after {} iterations", iteration + 1);
            return Ok((new_params, new_ll));
        }
        
        // Check for likelihood decrease (shouldn't happen in correct EM)
        if ll_change < -0.1 {
            println!("Warning: Log-likelihood decreased by {:.4}", ll_change);
        }
        
        params = new_params;
        prev_ll = new_ll;
    }
    
    println!("Reached max iterations");
    Ok((params, prev_ll))
}