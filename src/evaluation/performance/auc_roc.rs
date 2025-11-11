use crate::{evaluation::{em_algorithm::{em_result::EmResult, formatted_record::FormattedRecord}, performance::{load_data::{load_data, load_students}, prediction::{self, Prediction}}}, models::{hidden_markov_model, knowledge_tracing_model, models::Models}};
use std::{collections::HashMap, error::Error};
pub async fn benchmark_model_with_auc(model: Models, initial_parameters: EmResult, input: &str) -> Result<(), Box<dyn Error>> {
    let records = load_data(input).await?;
    let mut users = load_students(&records, initial_parameters).await?;
    println!("Initialized {} students with skill maps.", users.len());
    let predictions = match model {
        Models::HiddenMarkovModel => {
            evaluate_hmm(&mut users, &records, initial_parameters.transition).await
        }
        Models::KnowledgeTracingModel => {
            evaluate_ktm(&mut users, &records, initial_parameters.transition, initial_parameters.slip, initial_parameters.guess).await
        }
    };

    Ok(())
}


async fn evaluate_hmm(users: &mut HashMap<u32, HashMap<u32, f64>>, records: &Vec<FormattedRecord>, transition: f64) ->  Vec<Prediction> {
    let mut predictions = Vec::new();
    for record in records {
        if let Some(skill_map) = users.get_mut(&record.user_id) {
            if let Some(prob) = skill_map.get_mut(&record.skill_id) {
                predictions.push(Prediction {
                    probability: *prob,
                    actual: record.correct == 1
                });
                let new_prob = hidden_markov_model::calculate_mastery(*prob, transition).await;
                *prob = new_prob;
            }
        }
    }
    predictions
}

async fn evaluate_ktm(users: &mut HashMap<u32, HashMap<u32, f64>>,records: &Vec<FormattedRecord>, transition: f64, slip: f64, guess: f64) -> Vec<Prediction> {
    let mut predictions = Vec::new();

    for record in records {
        if let Some(skill_map) = users.get_mut(&record.user_id) {
            if let Some(prob) = skill_map.get_mut(&record.skill_id) {
                predictions.push(Prediction {
                    probability: *prob,
                    actual: record.correct == 1,
                });
                let new_prob = knowledge_tracing_model::calculate_mastery(
                    *prob,
                    transition,
                    slip,
                    guess,
                    record.correct == 1,
                )
                .await;
                *prob = new_prob;
            }
        }
    }

   predictions
}

