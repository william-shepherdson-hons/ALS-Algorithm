use crate::models::knowledge_tracing_model;
use crate::{evaluation::em_result::EmResult, models::hidden_markov_model};
use crate::models::models::Models;
use crate::evaluation::formatted_record::FormattedRecord;
use std::{collections::HashMap, error::Error};
use csv::{ReaderBuilder};

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

async fn forward_pass(observations: &[bool], params: &EmResult, model: &Models) -> Vec<f64> {
    let mut mastery_probs = Vec::with_capacity(observations.len() + 1);
    mastery_probs.push(params.initial);
    for &observed_correct in observations {
        let current_mastery = *mastery_probs.last().unwrap();

        let next_mastery = match model {
            Models::HiddenMarkovModel => {
                hidden_markov_model::calculate_mastery(current_mastery, params.transition).await
            },
            Models::KnowledgeTracingModel => {
                knowledge_tracing_model::calculate_mastery(current_mastery, params.transition, params.slip, params.guess, observed_correct).await
            }
        };

        mastery_probs.push(next_mastery.min(1.0).max(0.0));

    }
    mastery_probs
}

async fn backwards_pass(observations: &[bool], params: &EmResult, model: &Models) -> Vec<f64> {
    let n = observations.len();
    let mut backward = vec![0.0; n + 1];

    backward[n] = 1.0;
    for t in (0..n).rev() {
        let current_backward = backward[t + 1];
        let next_observation = observations[t];
        let beta = match  model {
            Models::HiddenMarkovModel => {
                hidden_markov_model::calculate_backward_probability(current_backward, next_observation, params.slip, params.guess, params.transition).await
            }
            Models::KnowledgeTracingModel => {
                knowledge_tracing_model::calculate_backward_probability(current_backward, next_observation, params.slip, params.guess).await
            }
        };
        backward[t] = beta;
    }
    backward

}

async fn smooth_probabilities(forward: &[f64], backward: &[f64]) -> Vec<f64> {
    let n = forward.len();
    let mut smoothed = Vec::with_capacity(n);
    let mut normalizer = 0.0;
    
    for t in 0..n {
        let gamma = forward[t] * backward[t];
        smoothed.push(gamma);
        normalizer += gamma;
    }
    
    if normalizer > 0.0 {
        for gamma in smoothed.iter_mut() {
            *gamma /= normalizer;
        }
    }
    
    smoothed
} 

async fn calculate_transition_expectations(observations: &[bool], forward: &[f64], backward: &[f64], params: &EmResult, model: &Models) -> Vec<f64> {
    let n = observations.len();
    let mut xi_values: Vec<f64> = Vec::with_capacity(n);
    for t in 0..n {
        let xi = match model {
            Models::HiddenMarkovModel => {
                hidden_markov_model::calculate_transistion_expectation(forward[t], backward[t+1], observations[t], params.transition, params.slip).await
            },
            Models::KnowledgeTracingModel => {
                knowledge_tracing_model::calculate_transition_expectation(forward[t], forward[t+1]).await
            }
        };
        xi_values.push(xi);
    }

    xi_values
}

async fn accumulate_sequence_counts(observations: &[bool], mastery_probs: &[f64], counts: &mut ExpectedCounts){
    counts.sum_initial_mastery += mastery_probs[0];
    counts.n_sequences += 1;
    for t in 0..observations.len() {
        let p_known_before = mastery_probs[t];
        let p_known_after = mastery_probs[t + 1];
        let p_unknown_before = 1.0 - p_known_before;
        counts.sum_learned += (p_known_after - p_known_before).max(0.0);
        counts.sum_opportunities_unknown += p_unknown_before;

        if observations[t] {
            counts.sum_correct_while_known += p_known_before;
            counts.sum_correct_while_unknown += p_unknown_before;
        }
        counts.sum_known += p_known_before;
        counts.sum_unknown += p_unknown_before;
    }
}

fn m_step_update(counts: &ExpectedCounts) -> EmResult {
    let initial = if counts.n_sequences > 0 {
        (counts.sum_initial_mastery / counts.n_sequences as f64)
            .min(0.99).max(0.01)
    } else {
        0.5
    };
    
    let transition = if counts.sum_opportunities_unknown > 0.0 {
        (counts.sum_learned / counts.sum_opportunities_unknown)
            .min(0.99).max(0.01)
    } else {
        0.1
    };
    
    let slip = if counts.sum_known > 0.0 {
        (1.0 - counts.sum_correct_while_known / counts.sum_known)
            .min(0.99).max(0.01)
    } else {
        0.1
    };
    
    let guess = if counts.sum_unknown > 0.0 {
        (counts.sum_correct_while_unknown / counts.sum_unknown)
            .min(0.99).max(0.01)
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
        initial: initial,
        transition: transition,
        slip: slip,
        guess: guess,
    }
}


pub async fn expectation_maximisation(model: Models, initial: EmResult, path: &str) -> Result<EmResult, Box<dyn Error>>{
    let mut reader = ReaderBuilder::new()
        .trim(csv::Trim::All)
        .from_path(path)?;
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

    println!("Starting EM with {} sequences", sequences.len());
    println!("Initial params: L0={:.4}, T={:.4}, S={:.4}, G={:.4}", 
             params.initial, params.transition, params.slip, params.guess);


    for iteration in 0..MAX_ITERATIONS {
        let mut counts = ExpectedCounts::new();
        for (_key, sequence) in &sequences {
            let mastery_probs = forward_pass(&sequence.observations, &params, &model).await;
            accumulate_sequence_counts(&sequence.observations, &mastery_probs, &mut counts).await
        }
        let new_params = m_step_update(&counts);

        let diff = (new_params.initial - params.initial).abs()
            + (new_params.transition - params.transition).abs()
            + (new_params.slip - params.slip).abs()
            + (new_params.guess - params.guess).abs();
        println!("Iteration {}: L0={:.4}, T={:.4}, S={:.4}, G={:.4}, diff={:.6}",
            iteration, new_params.initial, new_params.transition,
            new_params.slip, new_params.guess, diff);
        params = new_params;
        if diff < TOLERANCE {
            println!("Converged after {} iterations", iteration + 1);
            break;
        }
    }
    

    Ok(params)

}