use std::collections::HashMap;
use std::error::Error;
use chrono::{DateTime, NaiveDateTime, Utc};
use serde::{Deserialize, Serialize};
use csv::{Writer, ReaderBuilder};

#[derive(Debug, Deserialize, Serialize, Clone)]
struct Record {
    user_id: String,
    correct: String, 
    start_time: String,
    skill_id: String
}

#[derive(Debug, Serialize)]
struct FormattedRecord {
    user_id: String,
    correct: String,
    times_applied: u32,
    skill_id: String,
}

pub fn process_assistments() -> Result<(), Box<dyn Error>> {
    println!("Removing fields not associated with skills");
    filter_skills()?;

    println!("Sorting chronologically");
    chronological_order()?;

    println!("Formatting output");
    format_data()?;
    Ok(())
}

fn parse_time(s: &str) -> NaiveDateTime {
    NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S")
        .unwrap_or_else(|_| {
            DateTime::<Utc>::from_timestamp(0, 0)
                .unwrap()
                .naive_utc()
        })
}

fn chronological_order() -> Result<(), Box<dyn Error>> {
    println!("Sorting filtered CSV by user_id and start_time");

    let mut reader = ReaderBuilder::new()
        .trim(csv::Trim::All)
        .from_path("src/data/only_skill_questions.csv")?;

    let mut records: Vec<Record> = reader.deserialize().collect::<Result<_, _>>()?;

    // Sort by user_id and then by start_time
    records.sort_by_key(|r| (r.user_id.clone(), parse_time(&r.start_time)));

    let mut writer = Writer::from_path("src/data/chronological_skill_questions.csv")?;
    for record in records {
        writer.serialize(record)?;
    }

    writer.flush()?;
    println!("Chronological CSV saved to src/data/chronological_skill_questions.csv");
    Ok(())
}

fn filter_skills() -> Result<(), Box<dyn Error>> {
    println!("Filtering dataset to include only skill questions");

    let mut reader = ReaderBuilder::new()
        .trim(csv::Trim::All)
        .from_path("src/data/2012-2013-data-with-predictions-4-final.csv")?;

    let mut writer = Writer::from_path("src/data/only_skill_questions.csv")?;

    let mut total_rows = 0;
    let mut success_rows = 0;
    let mut no_skill = 0;
    let mut failed_rows = 0;

    for result in reader.deserialize::<Record>() { 
        total_rows += 1;
        match result {
            Ok(record) => {
                if record.skill_id.trim().is_empty() {
                    no_skill += 1;
                    continue;
                }
                writer.serialize(&record)?; 
                success_rows += 1;
            }
            Err(e) => {
                eprintln!("Failed to parse row {}: {}", total_rows, e);
                failed_rows += 1;
            }
        }
    }

    writer.flush()?;
    println!("Processed CSV saved to src/data/only_skill_questions.csv");
    println!(
        "Summary: total = {}, success = {}, failed = {}, skipped = {}",
        total_rows, success_rows, failed_rows, no_skill
    );

    Ok(())
}

fn format_data() -> Result<(), Box<dyn Error>> {
    println!("Formatting data and adding 'times_applied' column");

    let mut reader = ReaderBuilder::new()
        .trim(csv::Trim::All)
        .from_path("src/data/chronological_skill_questions.csv")?;

    let records: Vec<Record> = reader.deserialize().collect::<Result<_, _>>()?;

    let mut writer = Writer::from_path("src/data/final_formatted_skill_data.csv")?;

    let mut usage_count: HashMap<(String, String), u32> = HashMap::new();

    for record in records {
        let key = (record.user_id.clone(), record.skill_id.clone());
        let counter = usage_count.entry(key.clone()).or_insert(0);
        *counter += 1;

        let formatted = FormattedRecord {
            user_id: record.user_id,
            correct: record.correct,
            times_applied: *counter,
            skill_id: record.skill_id,
        };

        writer.serialize(formatted)?;
    }

    writer.flush()?;
    println!("Final formatted CSV saved to src/data/final_formatted_skill_data.csv");
    Ok(())
}
