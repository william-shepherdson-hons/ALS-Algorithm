use crate::evaluation::em_result::EmResult;
use crate::models::models::Models;
use crate::evaluation::formatted_record::FormattedRecord;
use std::{collections::HashMap, error::Error};
use serde::{Deserialize, Serialize};
use csv::{ReaderBuilder};

struct UserSkillSequence {
    observations: Vec<bool>,
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

    Ok(EmResult {
        guess: 0.0,
        transition: 0.0,
        initial: 0.0,
        slip: 0.0
    })
}