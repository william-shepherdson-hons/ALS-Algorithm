use crate::{evaluation::{em_algorithm::{em_result::EmResult, formatted_record::FormattedRecord}, performance::{load_data::{load_data, load_students}, prediction::{Prediction}}}, models::{hidden_markov_model, knowledge_tracing_model, models::Models}};
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

    let auc = calculate_auc_roc(predictions.clone());
    let roc_points = calculate_roc_curve(predictions);

    println!("\n=== Model Evaluation Results ===");
    println!("Model: {:?}", model);
    println!("AUC-ROC: {:.4}", auc);
    println!("Total predictions: {}", roc_points.len() - 1);

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

fn calculate_auc_roc(mut predictions: Vec<Prediction>) -> f64 {
    if predictions.is_empty() {
        return 0.5
    }
    predictions.sort_by(|a, b| b.probability.partial_cmp(&a.probability).unwrap());

    let total_positives = predictions.iter().filter(|p| p.actual).count() as f64;
    let total_negatives = predictions.len() as f64 - total_positives;

    if total_positives == 0.0 || total_negatives == 0.0 {
        return 0.5;
    }

    let mut tp = 0.0;
    let mut fp = 0.0;
    let mut auc = 0.0;
    let mut prev_fp = 0.0;

    for pred in predictions {
        if pred.actual {
            tp += 1.0;
        } else {
            fp += 1.0;

            auc += (fp - prev_fp) * tp;
            prev_fp = fp;
        }
    }

    auc / (total_positives * total_negatives)


}
fn calculate_roc_curve(mut predictions: Vec<Prediction>) -> Vec<(f64, f64)> {
    if predictions.is_empty() {
        return vec![(0.0, 0.0), (1.0, 1.0)];
    }

    predictions.sort_by(|a, b| b.probability.partial_cmp(&a.probability).unwrap());

    let total_positives = predictions.iter().filter(|p| p.actual).count() as f64;
    let total_negatives = predictions.len() as f64 - total_positives;

    if total_positives == 0.0 || total_negatives == 0.0 {
        return vec![(0.0, 0.0), (1.0, 1.0)];
    }

    let mut roc_points = vec![(0.0, 0.0)];
    let mut tp = 0.0;
    let mut fp = 0.0;

    for pred in predictions {
        if pred.actual {
            tp += 1.0;
        } else {
            fp += 1.0;
        }
        let tpr = tp / total_positives;
        let fpr = fp / total_negatives;
        roc_points.push((fpr, tpr));
    }

    roc_points
}