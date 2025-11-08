use crate::evaluation::em_result::EmResult;
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

async fn forward_pass(observations: &[bool], params: &EmResult) -> Vec<f64> {
    let mut mastery_probs = Vec::with_capacity(observations.len() + 1);
    mastery_probs.push(params.initial);
    mastery_probs
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
    const MAX_ITERATIONS: usize = 100;
    const TOLERANCE: f64 = 1e-4;
    let mut params = EmResult {
        initial: initial.initial,
        transition: initial.transition,
        slip: initial.slip,
        guess: initial.guess,
    };

    println!("Starting EM with {} sequences", sequences.len());
    println!("Initial params: L0={:.4}, T={:.4}, S={:.4}, G={:.4}", 
             params.initial, params.transition, params.slip, params.guess);


    for iteration in 0..MAX_ITERATIONS {
        let mut counts = ExpectedCounts::new();
    }

    Ok(EmResult {
        guess: 0.0,
        transition: 0.0,
        initial: 0.0,
        slip: 0.0
    })

}