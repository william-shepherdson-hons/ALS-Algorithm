use crate::evaluation::em_result::EmResult;
use crate::models::models::Models;
use crate::evaluation::formatted_record::FormattedRecord;
use std::error::Error;
use serde::{Deserialize, Serialize};
use csv::{ReaderBuilder};

pub async fn expectation_maximisation(model: Models, initial: EmResult, path: &str) -> Result<EmResult, Box<dyn Error>>{
    let mut reader = ReaderBuilder::new()
        .trim(csv::Trim::All)
        .from_path(path)?;
    let records: Vec<FormattedRecord> = reader.deserialize().collect::<Result<_, _>>()?;

    
    Ok(EmResult {
        guess: 0.0,
        transition: 0.0,
        initial: 0.0,
        slip: 0.0
    })
}