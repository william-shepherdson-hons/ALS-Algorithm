use std::collections::{HashSet,HashMap};

use std::error::Error;
use rand::rng;
use rand::seq::SliceRandom;
use chrono::{DateTime, NaiveDateTime, Utc};
use serde::{Deserialize, Serialize};
use csv::{Writer, ReaderBuilder};
use crate::evaluation::em_algorithm::formatted_record::FormattedRecord;

#[derive(Debug, Deserialize, Serialize, Clone)]
struct Record {
    user_id: String,
    correct: String, 
    start_time: String,
    skill_id: String
}



pub fn process_assistments() -> Result<(), Box<dyn Error>> {
    println!("Removing fields not associated with skills");
    filter_skills()?;

    println!("Sorting chronologically");
    chronological_order()?;

    println!("Formatting output");
    format_data()?;
    split_data(20)?;
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
        let correct_value = record.correct.trim().parse::<u32>().unwrap_or(0);
        let user_id_value= record.user_id.trim().parse::<u32>().unwrap_or(0);
        let skill_id_value = record.skill_id.trim().parse::<u32>().unwrap_or(0);

        let formatted = FormattedRecord {
            user_id: user_id_value,
            correct: correct_value,
            times_applied: *counter,
            skill_id: skill_id_value,
        };

        writer.serialize(formatted)?;
    }

    writer.flush()?;
    println!("Final formatted CSV saved to src/data/final_formatted_skill_data.csv");
    Ok(())
}




fn split_data(test_split: i32) -> Result<(), Box<dyn Error>> {
    println!("Splitting data by students ({}% test)", test_split);

    // Read the final formatted CSV
    let mut reader = ReaderBuilder::new()
        .trim(csv::Trim::All)
        .from_path("src/data/final_formatted_skill_data.csv")?;

    let records: Vec<FormattedRecord> = reader.deserialize().collect::<Result<_, _>>()?;
    println!("Loaded {} total records", records.len());

    // Collect all unique user_ids
    let mut users: Vec<u32> = records
        .iter()
        .map(|r| r.user_id.clone())
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();

    println!("Found {} unique users", users.len());

    // Shuffle users for random splitting
    let mut rng = rng();
    users.shuffle(&mut rng);

    // Determine split index for users
    let test_user_count = (users.len() as f32 * (test_split as f32 / 100.0)).round() as usize;
    let (test_users, train_users) = users.split_at(test_user_count);

    let test_user_set: HashSet<u32> = test_users.iter().cloned().collect();
    let train_user_set: HashSet<u32> = train_users.iter().cloned().collect();

    // Split records by user_id
    let mut train_records = Vec::new();
    let mut test_records = Vec::new();

    for record in records {
        if test_user_set.contains(&record.user_id) {
            test_records.push(record);
        } else if train_user_set.contains(&record.user_id) {
            train_records.push(record);
        }
    }

    println!(
        "Split complete: {} train users, {} test users",
        train_user_set.len(),
        test_user_set.len()
    );
    println!(
        "Train records: {}, Test records: {}",
        train_records.len(),
        test_records.len()
    );

    let mut train_writer = Writer::from_path("src/data/train_data.csv")?;
    for record in &train_records {
        train_writer.serialize(record)?;
    }
    train_writer.flush()?;
    println!("Training data saved to src/data/train_data.csv");

    let mut test_writer = Writer::from_path("src/data/test_data.csv")?;
    for record in &test_records {
        test_writer.serialize(record)?;
    }
    test_writer.flush()?;
    println!("Test data saved to src/data/test_data.csv");

    Ok(())
}

