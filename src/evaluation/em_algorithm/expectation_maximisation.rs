use crate::evaluation::em_algorithm::iteration::Iteration;
use crate::models::knowledge_tracing_model;
use crate::{evaluation::em_algorithm::em_result::EmResult, models::hidden_markov_model};
use crate::models::models::Models;
use crate::evaluation::em_algorithm::formatted_record::FormattedRecord;
use std::{collections::HashMap, error::Error};
use csv::{Writer,ReaderBuilder};


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

/// Forward pass (kept original structure using calculate_mastery)
async fn forward_pass(observations: &[bool], params: &EmResult, model: &Models) -> Vec<f64> {
    let mut mastery_probs = Vec::with_capacity(observations.len() + 1);
    mastery_probs.push(params.initial);

    for &observed_correct in observations {
        let current_mastery = *mastery_probs.last().unwrap();
        let next_mastery = match model {
            Models::HiddenMarkovModel => {
                hidden_markov_model::calculate_mastery(current_mastery, params.transition).await
            }
            Models::KnowledgeTracingModel => {
                knowledge_tracing_model::calculate_mastery(
                    current_mastery,
                    params.transition,
                    params.slip,
                    params.guess,
                    observed_correct,
                )
                .await
            }
        };
        mastery_probs.push(next_mastery.min(1.0).max(0.0));
    }
    mastery_probs
}

/// Fixed backward pass: computes backward_known and backward_unknown for 2-state model
async fn backwards_pass(
    observations: &[bool],
    params: &EmResult,
    model: &Models,
) -> (Vec<f64>, Vec<f64>) {
    let n = observations.len();
    let mut backward_known = vec![0.0; n + 1];
    let mut backward_unknown = vec![0.0; n + 1];
    backward_known[n] = 1.0;
    backward_unknown[n] = 1.0;

    let emission = |state_known: bool, obs: bool, p: &EmResult| -> f64 {
        if obs {
            if state_known {
                1.0 - p.slip
            } else {
                p.guess
            }
        } else {
            if state_known {
                p.slip
            } else {
                1.0 - p.guess
            }
        }
    };

    for t in (0..n).rev() {
        let obs_next = observations[t];

        match model {
            Models::HiddenMarkovModel => {
                let p_kk = params.transition;
                let p_ku = 1.0 - params.transition;
                let p_uk = params.transition;
                let p_uu = 1.0 - params.transition;

                let e_known = emission(true, obs_next, params);
                let e_unknown = emission(false, obs_next, params);

                backward_known[t] = p_kk * e_known * backward_known[t + 1]
                    + p_ku * e_unknown * backward_unknown[t + 1];
                backward_unknown[t] = p_uk * e_known * backward_known[t + 1]
                    + p_uu * e_unknown * backward_unknown[t + 1];
            }
            Models::KnowledgeTracingModel => {
                let p_kk = 1.0;
                let p_ku = 0.0;
                let p_uk = params.transition;
                let p_uu = 1.0 - params.transition;

                let e_known = emission(true, obs_next, params);
                let e_unknown = emission(false, obs_next, params);

                backward_known[t] = p_kk * e_known * backward_known[t + 1]
                    + p_ku * e_unknown * backward_unknown[t + 1];
                backward_unknown[t] = p_uk * e_known * backward_known[t + 1]
                    + p_uu * e_unknown * backward_unknown[t + 1];
            }
        }

        backward_known[t] = backward_known[t].max(0.0);
        backward_unknown[t] = backward_unknown[t].max(0.0);
    }

    (backward_known, backward_unknown)
}

/// Per-time-step posterior P(known_t | all obs)
fn smooth_probabilities(
    forward: &[f64],
    backward_known: &[f64],
    backward_unknown: &[f64],
) -> Vec<f64> {
    let n = forward.len();
    let mut smoothed = Vec::with_capacity(n);

    for t in 0..n {
        let alpha_known = forward[t];
        let beta_known = backward_known[t];
        let alpha_unknown = 1.0 - alpha_known;
        let beta_unknown = backward_unknown[t];

        let num_known = alpha_known * beta_known;
        let num_unknown = alpha_unknown * beta_unknown;
        let denom = num_known + num_unknown;

        smoothed.push(if denom > 0.0 {
            num_known / denom
        } else {
            alpha_known
        });
    }
    smoothed
}

/// Compute expected number of unknown→known transitions per time step
async fn calculate_transition_expectations(
    observations: &[bool],
    forward: &[f64],
    backward_known: &[f64],
    backward_unknown: &[f64],
    params: &EmResult,
    model: &Models,
) -> Vec<f64> {
    let n = observations.len();
    let mut xi_values: Vec<f64> = Vec::with_capacity(n);

    let emission = |state_known: bool, obs: bool, p: &EmResult| -> f64 {
        if obs {
            if state_known {
                1.0 - p.slip
            } else {
                p.guess
            }
        } else {
            if state_known {
                p.slip
            } else {
                1.0 - p.guess
            }
        }
    };

    for t in 0..n {
        let alpha_known_t = forward[t];
        let alpha_unknown_t = 1.0 - alpha_known_t;
        let obs_tp1 = observations[t];
        let e_known = emission(true, obs_tp1, params);
        let e_unknown = emission(false, obs_tp1, params);

        let (p_uk, p_uu, p_kk, p_ku) = match model {
            Models::HiddenMarkovModel => {
                let p_kk = params.transition;
                let p_ku = 1.0 - params.transition;
                let p_uk = params.transition;
                let p_uu = 1.0 - params.transition;
                (p_uk, p_uu, p_kk, p_ku)
            }
            Models::KnowledgeTracingModel => {
                let p_kk = 1.0;
                let p_ku = 0.0;
                let p_uk = params.transition;
                let p_uu = 1.0 - params.transition;
                (p_uk, p_uu, p_kk, p_ku)
            }
        };

        let bk_tp1 = backward_known[t + 1];
        let bu_tp1 = backward_unknown[t + 1];

        let j_kk = alpha_known_t * p_kk * e_known * bk_tp1;
        let j_uk = alpha_unknown_t * p_uk * e_known * bk_tp1;
        let j_uu = alpha_unknown_t * p_uu * e_unknown * bu_tp1;
        let j_ku = alpha_known_t * p_ku * e_unknown * bu_tp1;

        let denom = j_kk + j_uk + j_uu + j_ku;
        let xi = if denom > 0.0 { j_uk / denom } else { 0.0 };

        xi_values.push(xi.max(0.0));
    }

    xi_values
}

/// Accumulate expected counts for one user-skill sequence
async fn accumulate_sequence_counts(
    observations: &[bool],
    forward: &[f64],
    backward_known: &[f64],
    backward_unknown: &[f64],
    params: &EmResult,
    model: &Models,
    counts: &mut ExpectedCounts,
) {
    let smoothed = smooth_probabilities(forward, backward_known, backward_unknown);

    counts.sum_initial_mastery += smoothed[0];
    counts.n_sequences += 1;

    let xi_values = calculate_transition_expectations(
        observations,
        forward,
        backward_known,
        backward_unknown,
        params,
        model,
    )
    .await;

    for t in 0..observations.len() {
        let p_known = smoothed[t];
        let p_unknown = 1.0 - p_known;

        counts.sum_learned += xi_values[t];
        counts.sum_opportunities_unknown += p_unknown;

        if observations[t] {
            counts.sum_correct_while_known += p_known;
            counts.sum_correct_while_unknown += p_unknown;
        }

        counts.sum_known += p_known;
        counts.sum_unknown += p_unknown;
    }
}

/// Maximization step — unchanged
fn m_step_update(counts: &ExpectedCounts) -> EmResult {
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

/// Main EM loop
pub async fn expectation_maximisation(
    model: Models,
    initial: EmResult,
    path: &str,
    output: &str
) -> Result<EmResult, Box<dyn Error>> {
    let mut reader = ReaderBuilder::new().trim(csv::Trim::All).from_path(path)?;
    let mut writer = Writer::from_path(output)?;
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

    const MAX_ITERATIONS: usize = 1000;
    const TOLERANCE: f64 = 1e-4;
    let mut params = initial;

    println!(
        "Starting EM with {} sequences",
        sequences.len()
    );
    println!(
        "Initial params: L0={:.4}, T={:.4}, S={:.4}, G={:.4}",
        params.initial, params.transition, params.slip, params.guess
    );

    for iteration in 0..MAX_ITERATIONS {
        let mut counts = ExpectedCounts::new();

        for (_key, sequence) in &sequences {
            let forward = forward_pass(&sequence.observations, &params, &model).await;
            let (backward_known, backward_unknown) =
                backwards_pass(&sequence.observations, &params, &model).await;
            accumulate_sequence_counts(
                &sequence.observations,
                &forward,
                &backward_known,
                &backward_unknown,
                &params,
                &model,
                &mut counts,
            )
            .await;
        }

        let new_params = m_step_update(&counts);

        let diff = (new_params.initial - params.initial).abs()
            + (new_params.transition - params.transition).abs()
            + (new_params.slip - params.slip).abs()
            + (new_params.guess - params.guess).abs();

        
        let iteration_output = Iteration {
            iteration: iteration,
            initial: new_params.initial,
            transition: new_params.transition,
            slip: new_params.slip,
            guess: new_params.guess,
            diff: diff
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
