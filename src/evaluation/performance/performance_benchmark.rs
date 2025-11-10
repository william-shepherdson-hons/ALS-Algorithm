use std::{collections::HashMap, time::Duration};
use std::error::Error;
use std::time::Instant;
use csv::ReaderBuilder;

use crate::{
    evaluation::em_algorithm::{em_result::EmResult, formatted_record::FormattedRecord},
    models::hidden_markov_model,
    models::knowledge_tracing_model,
    models::models::Models,
};



async fn benchmark_hmm(users: &mut HashMap<u32, HashMap<u32, f64>>, records: &Vec<FormattedRecord>, transition: f64) -> Duration {
    println!("Starting HMM benchmark on {} records", records.len());
    let now = Instant::now();
    for record in records {
        if let Some(skill_map) = users.get_mut(&record.user_id) {
            if let Some(prob) = skill_map.get_mut(&record.skill_id) {
                let new_prob = hidden_markov_model::calculate_mastery(*prob, transition).await;
                *prob = new_prob;
            }
        }
    }
    now.elapsed()

}

async fn benchmark_ktm(users: &mut HashMap<u32, HashMap<u32, f64>>, records: &Vec<FormattedRecord>, transition: f64, slip: f64, guess:f64) -> Duration {
    println!("Starting KTM benchmark on {} records", records.len());
    let now = Instant::now();
    for record in records {
        if let Some(skill_map) = users.get_mut(&record.user_id) {
            if let Some(prob) = skill_map.get_mut(&record.skill_id) {
                let new_prob = knowledge_tracing_model::calculate_mastery(*prob, transition, slip, guess, record.correct == 1).await;
                *prob = new_prob;
            }
        }
    }
    now.elapsed()
}


async fn load_data(input: &str) -> Result<Vec<FormattedRecord>, Box<dyn  Error>> {
    println!("Starting data loading benchmark");
    let now = Instant::now();

    let mut reader = ReaderBuilder::new()
        .trim(csv::Trim::All)
        .from_path(input)?;
    let records: Vec<FormattedRecord> = reader.deserialize().collect::<Result<_, _>>()?;

    let elapsed = now.elapsed();
    println!("Data Loading: {:?} Loaded: {:?} records", elapsed, records.len());

    Ok(records)
}

pub async fn benchmark_model_performance(model: Models, initial_parameters: EmResult, input: &str, iterations: usize) -> Result<(), Box<dyn Error>> {
    let records = load_data(input).await?;
    let mut users: HashMap<u32, HashMap<u32, f64>> = HashMap::new();

    for record in &records {
        let skill_map = users
            .entry(record.user_id)
            .or_insert_with(HashMap::new);
        skill_map
            .entry(record.skill_id)
            .or_insert(initial_parameters.initial);
    }
    println!("Initialized {} students with skill maps.", users.len());

    Ok(())
}

async fn benchmark_model_performance_single(model: Models, initial_parameters: EmResult, records: &Vec<FormattedRecord>, users: &mut HashMap<u32, HashMap<u32, f64>>) -> Result<Duration, Box<dyn Error>> {
    Ok( match model {
        Models::HiddenMarkovModel => benchmark_hmm(users, &records, initial_parameters.transition).await,
        Models::KnowledgeTracingModel => benchmark_ktm(users, &records, initial_parameters.transition, initial_parameters.slip, initial_parameters.guess).await,
    })
}
