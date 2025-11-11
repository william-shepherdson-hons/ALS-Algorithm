use std::error::Error;
use std::time::Instant;
use csv::ReaderBuilder;

use crate::{
    evaluation::em_algorithm::{ formatted_record::FormattedRecord},

};


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