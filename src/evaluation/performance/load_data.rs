use std::error::Error;
use std::time::Instant;
use csv::ReaderBuilder;
use std::collections::HashMap;

use crate::evaluation::em_algorithm::{ em_result::EmResult, formatted_record::FormattedRecord};


pub async fn load_data(input: &str) -> Result<Vec<FormattedRecord>, Box<dyn  Error>> {
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

pub async fn load_students(records: &Vec<FormattedRecord>, initial_parameters: EmResult) -> Result<HashMap<u32, HashMap<u32, f64>>, Box<dyn  Error>> {
    let mut users: HashMap<u32, HashMap<u32, f64>> = HashMap::new();
    for record in records {
        let skill_map = users
            .entry(record.user_id)
            .or_insert_with(HashMap::new);
        skill_map
            .entry(record.skill_id)
            .or_insert(initial_parameters.initial);
    }
    Ok(users)
}