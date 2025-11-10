use std::time::Instant;
use csv::ReaderBuilder;
use std::error::Error;
use std::iter::Map;

use crate::{evaluation::em_algorithm::{em_result::EmResult, formatted_record::FormattedRecord}, models::models::Models};
struct User {
     user_id: usize,
     skills: Map<usize, f64>
}

fn benchmark_hmm(students: &Vec<User>, records: &Vec<FormattedRecord>) {

}
fn benchmark_ktm(students: &Vec<User>, records: &Vec<FormattedRecord>) {
    
}

pub async fn benchmark_model_performance(model: Models, initial_parameters: EmResult, input: &str) -> Result<(), Box<dyn Error>>{
    let now = Instant::now();
    let mut reader = ReaderBuilder::new()
        .trim(csv::Trim::All)
        .from_path(input)?;
    let records: Vec<FormattedRecord>;
    {
        records = reader.deserialize().collect::<Result<_, _>>()?;
    }

    let elapsed = now.elapsed();
    println!("Data Loading: {:?} Loaded: {:?} records", elapsed, records.len());

    Ok(())
}