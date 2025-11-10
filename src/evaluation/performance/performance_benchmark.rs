use std::collections::HashMap;
use std::error::Error;
use std::time::Instant;
use csv::ReaderBuilder;

use crate::{
    evaluation::em_algorithm::{em_result::EmResult, formatted_record::FormattedRecord},
    models::models::Models,
};

#[derive(Debug)]
struct User {
    user_id: usize,
    skills: HashMap<usize, f64>,
}

fn benchmark_hmm(students: &HashMap<usize, User>, records: &Vec<FormattedRecord>) {
    
}

fn benchmark_ktm(students: &HashMap<usize, User>, records: &Vec<FormattedRecord>) {
    
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
    let mut users: HashMap<usize, User> = HashMap::new();

    for record in &records {
        let user = users
            .entry(record.user_id as usize)
            .or_insert_with(|| User {
                user_id: record.user_id as usize,
                skills: HashMap::new(),
            });

        user.skills
            .entry(record.skill_id as usize)
            .or_insert(initial_parameters.initial);
    }

    println!("Initialized {} students with skill maps.", users.len());

    match model {
        Models::HiddenMarkovModel => benchmark_hmm(&users, &records),
        Models::KnowledgeTracingModel => benchmark_ktm(&users, &records),
        _ => println!("Unsupported model type for benchmarking."),
    }

    Ok(())
}
