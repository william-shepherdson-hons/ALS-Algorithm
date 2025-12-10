use csv::Writer;

use crate::{evaluation::{em_algorithm::{em_result::EmResult, formatted_record::FormattedRecord}, performance::{load_data::{load_data, load_students}, prediction::{Prediction}}}, models::{hidden_markov_model, knowledge_tracing_model, models::Models}};
use std::{collections::HashMap, error::Error};
pub async fn benchmark_model_with_metrics(model: Models, initial_parameters: EmResult, input: &str) -> Result<(), Box<dyn Error>> {
    let records = load_data(input).await?;
    let mut users = load_students(&records, initial_parameters).await?;
    println!("Initialized {} students with skill maps.", users.len());
    let predictions = match model {
        Models::HiddenMarkovModel => {
            evaluate_hmm(&mut users, &records, initial_parameters.transition, initial_parameters.slip, initial_parameters.guess).await
        }
        Models::KnowledgeTracingModel => {
            evaluate_ktm(&mut users, &records, initial_parameters.transition, initial_parameters.slip, initial_parameters.guess).await
        }
    };

    let auc = calculate_auc_roc(predictions.clone());
    let roc_points = calculate_roc_curve(predictions.clone());

    println!("\n=== Model Evaluation Results ===");
    println!("Model: {:?}", model);
    println!("AUC-ROC: {:.4}", auc);
    println!("Total predictions: {}", roc_points.len() - 1);

    let metrics = calculate_classification_metrics(&predictions, 0.5);

    println!("Accuracy: {:.4}", metrics["accuracy"]);
    println!("Precision: {:.4}", metrics["precision"]);
    println!("Recall: {:.4}", metrics["recall"]);
    println!("Specificity: {:.4}", metrics["specificity"]);
    println!("F1 Score: {:.4}", metrics["f1_score"]);
    println!("Log Loss: {:.4}", metrics["log_loss"]);
    println!("Brier Score: {:.4}", metrics["brier_score"]);

    let path = format!("src/data/{:?}_auc_roc.csv", model);

    let mut writer = Writer::from_path(path)?;
    writer.write_record(&["fpr", "tpr"])?;
    for (fpr, tpr) in roc_points {
        writer.write_record(&[&fpr.to_string(), &tpr.to_string()])?;
    }
    Ok(())
}


pub async fn evaluate_hmm(users: &mut HashMap<u32, HashMap<u32, f64>>, records: &Vec<FormattedRecord>, transition: f64, slip: f64, guess: f64) ->  Vec<Prediction> {
    let mut predictions = Vec::new();
    for record in records {
        if let Some(skill_map) = users.get_mut(&record.user_id) {
            if let Some(prob) = skill_map.get_mut(&record.skill_id) {
                let p_correct = hidden_markov_model::calculate_success(*prob, slip, guess).await;
                predictions.push(Prediction {
                    probability: p_correct,
                    actual: record.correct == 1
                });
                let new_prob = hidden_markov_model::calculate_mastery(*prob, transition).await;
                *prob = new_prob;
            }
        }
    }
    predictions
}

pub async fn evaluate_ktm(users: &mut HashMap<u32, HashMap<u32, f64>>,records: &Vec<FormattedRecord>, transition: f64, slip: f64, guess: f64) -> Vec<Prediction> {
    let mut predictions = Vec::new();

    for record in records {
        if let Some(skill_map) = users.get_mut(&record.user_id) {
            if let Some(prob) = skill_map.get_mut(&record.skill_id) {
                let p_correct = knowledge_tracing_model::calculate_success(*prob, slip, guess).await;
                predictions.push(Prediction {
                    probability: p_correct,
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

pub fn calculate_auc_roc(mut predictions: Vec<Prediction>) -> f64 {
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

pub fn calculate_classification_metrics(predictions: &Vec<Prediction>, threshold: f64) -> HashMap<&'static str, f64> {
    let mut tp = 0.0;
    let mut tn = 0.0;
    let mut fp = 0.0;
    let mut fn_ = 0.0;
    let mut log_loss = 0.0;
    let mut brier = 0.0;

    for p in predictions {
        let predicted = p.probability >= threshold;
        if p.actual && predicted {
            tp += 1.0;
        } else if !p.actual && !predicted {
            tn += 1.0;
        } else if !p.actual && predicted {
            fp += 1.0;
        } else if p.actual && !predicted {
            fn_ += 1.0;
        }

        // For log loss and Brier score
        let y = if p.actual { 1.0 } else { 0.0 };
        let eps = 1e-15;
        let prob = p.probability.clamp(eps, 1.0 - eps);
        log_loss += -(y * prob.ln() + (1.0 - y) * (1.0 - prob).ln());
        brier += (prob - y).powi(2);
    }

    let total = tp + tn + fp + fn_;
    let accuracy = (tp + tn) / total;
    let precision = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
    let recall = if tp + fn_ > 0.0 { tp / (tp + fn_) } else { 0.0 };
    let specificity = if tn + fp > 0.0 { tn / (tn + fp) } else { 0.0 };
    let f1 = if precision + recall > 0.0 { 2.0 * precision * recall / (precision + recall) } else { 0.0 };

    HashMap::from([
        ("accuracy", accuracy),
        ("precision", precision),
        ("recall", recall),
        ("specificity", specificity),
        ("f1_score", f1),
        ("log_loss", log_loss / total),
        ("brier_score", brier / total),
    ])
}
