use std::{collections::HashMap, time::Duration};
use std::error::Error;
use std::time::Instant;

use crate::{
    evaluation::em_algorithm::{em_result::EmResult, formatted_record::FormattedRecord},
    models::hidden_markov_model,
    models::knowledge_tracing_model,
    models::models::Models,
    evaluation::performance::load_data::{load_data, load_students}
};



async fn benchmark_hmm(users: &mut HashMap<u32, HashMap<u32, f64>>, records: &Vec<FormattedRecord>, transition: f64) -> Duration {
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




pub async fn benchmark_model_performance(model: Models, initial_parameters: EmResult, input: &str, iterations: usize) -> Result<(), Box<dyn Error>> {
    let records = load_data(input).await?;
    let users = load_students(&records, initial_parameters).await?;
    println!("Initialized {} students with skill maps.", users.len());
    let mut total_time = Duration::new(0, 0);
    for _ in  0..iterations {
        total_time += benchmark_model_performance_single(model, initial_parameters, &records, &mut users.clone()).await?
    }
    let avg_time = total_time / iterations as u32;
    println!(
        "Completed {} iterations.\nAverage time per iteration: {:?}\nTotal time: {:?}",
        iterations, avg_time, total_time
    );


    Ok(())
}

async fn benchmark_model_performance_single(model: Models, initial_parameters: EmResult, records: &Vec<FormattedRecord>, users: &mut HashMap<u32, HashMap<u32, f64>>) -> Result<Duration, Box<dyn Error>> {
    Ok( match model {
        Models::HiddenMarkovModel => benchmark_hmm(users, &records, initial_parameters.transition).await,
        Models::KnowledgeTracingModel => benchmark_ktm(users, &records, initial_parameters.transition, initial_parameters.slip, initial_parameters.guess).await,
    })
}
