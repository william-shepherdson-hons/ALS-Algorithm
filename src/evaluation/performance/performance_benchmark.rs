use std::collections::HashMap;
use std::error::Error;
use std::time::Instant;
use csv::ReaderBuilder;

use crate::{
    evaluation::em_algorithm::{em_result::EmResult, formatted_record::FormattedRecord},
    models::hidden_markov_model,
    models::knowledge_tracing_model,
    models::models::Models,
};



async fn benchmark_hmm(users: &mut HashMap<u32, HashMap<u32, f64>>, records: &Vec<FormattedRecord>, transition: f64) {
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
    let elapsed = now.elapsed();
    println!("Finished HMM benchmark in {:?}", elapsed)
}

async fn benchmark_ktm(users: &HashMap<u32, HashMap<u32, f64>>, records: &Vec<FormattedRecord>) {
    
}

pub async fn benchmark_model_performance(model: Models, initial_parameters: EmResult, input: &str) -> Result<(), Box<dyn Error>> {
    let now = Instant::now();

    let mut reader = ReaderBuilder::new()
        .trim(csv::Trim::All)
        .from_path(input)?;
    let records: Vec<FormattedRecord> = reader.deserialize().collect::<Result<_, _>>()?;

    let elapsed = now.elapsed();
    println!("Data Loading: {:?} Loaded: {:?} records", elapsed, records.len());

    let mut student_map: HashMap<usize, HashMap<usize, f64>> = HashMap::new();

    for record in &records {
        let user_entry = student_map.entry(record.user_id as usize).or_insert_with(HashMap::new);
        user_entry.entry(record.skill_id as usize)
            .or_insert(initial_parameters.initial); 
    }
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

    match model {
        Models::HiddenMarkovModel => benchmark_hmm(&mut users, &records, initial_parameters.transition).await,
        Models::KnowledgeTracingModel => benchmark_ktm(&mut users, &records).await,
        _ => println!("Unsupported model type for benchmarking."),
    }

    Ok(())
}
