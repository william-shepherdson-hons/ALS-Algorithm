use std::error::Error;
use serde::{Deserialize, Serialize};
use csv::{Writer, ReaderBuilder};

#[derive(Debug, Deserialize, Serialize)]
struct Record {
    user_id: String,
    correct: String, 
    start_time: String,
    skill_id: String
}

pub fn process_assistments() -> Result<(), Box<dyn Error>> {
    println!("Removing Fields not associted with skills");
    _ = filter_skills();
    _ = chronological_order();
    
    Ok(())
}

fn chronological_order() -> Result<(), Box<dyn Error>>{

    Ok(())

}

fn filter_skills() -> Result<(), Box<dyn Error>>{
        println!("Processing dataset");

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
    println!("Processed CSV saved to processed_data.csv");
    println!(
        "Summary: total rows = {}, successfully processed = {}, failed = {}, Skipped = {}",
        total_rows, success_rows, failed_rows, no_skill
    );

    Ok(())
}
