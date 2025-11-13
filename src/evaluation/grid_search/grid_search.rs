use crate::{
    evaluation::{
        em_algorithm::{em_result::EmResult, formatted_record::FormattedRecord},
        performance::{
            load_data::{load_data, load_students},
            auc_roc::{evaluate_hmm, evaluate_ktm, calculate_auc_roc, calculate_classification_metrics}
        },
    },
    models::{models::Models},
};
use std::{error::Error};

#[derive(Debug, Clone)]
pub struct GridSearchResult {
    pub initial: f64,
    pub transition: f64,
    pub slip: f64,
    pub guess: f64,
    pub auc: f64,
    pub log_loss: f64,
    pub brier_score: f64,
    pub accuracy: f64,
    pub f1_score: f64,
}

impl GridSearchResult {
    fn print(&self, rank: usize) {
        println!("\n#{} - AUC: {:.4}", rank, self.auc);
        println!("  Initial: {:.3}, Transition: {:.3}, Slip: {:.3}, Guess: {:.3}",
                 self.initial, self.transition, self.slip, self.guess);
        println!("  Accuracy: {:.4}, F1: {:.4}, Log Loss: {:.4}, Brier: {:.4}",
                 self.accuracy, self.f1_score, self.log_loss, self.brier_score);
    }
}

pub async fn grid_search_hyperparameters(
    model: Models,
    input: &str,
    optimize_for: &str, // "auc", "log_loss", "f1_score"
) -> Result<Vec<GridSearchResult>, Box<dyn Error>> {
    println!("Starting hyperparameter grid search...");
    println!("Model: {:?}", model);
    println!("Optimizing for: {}\n", optimize_for);

    // Define parameter grid
    let initial_range = vec![0.05, 0.10, 0.15, 0.20, 0.25, 0.30,0.5];
    let transition_range = vec![0.05, 0.10, 0.15, 0.20, 0.25, 0.30,0.5];
    let slip_range = vec![0.05, 0.10, 0.15, 0.20,0.5];
    let guess_range = vec![0.10, 0.15, 0.20, 0.25, 0.30,0.5];

    let total_combinations = initial_range.len() 
        * transition_range.len() 
        * slip_range.len() 
        * guess_range.len();
    
    println!("Testing {} parameter combinations\n", total_combinations);

    let records = load_data(input).await?;
    let mut results = Vec::new();
    let mut tested = 0;

    for &initial in &initial_range {
        for &transition in &transition_range {
            for &slip in &slip_range {
                for &guess in &guess_range {
                    tested += 1;
                    
                    // Skip invalid parameter combinations
                    if slip + guess > 1.0 {
                        continue; // Unrealistic parameters
                    }

                    // Create parameter set
                    let params = EmResult {
                        initial,
                        transition,
                        slip,
                        guess,
                    };

                    // Evaluate model
                    match evaluate_parameters(&records, params, model).await {
                        Ok(result) => {
                            if tested % 50 == 0 {
                                let best_metric = get_best_metric(&results, optimize_for);
                                println!("Progress: {}/{} ({:.1}%) - Current best {}: {:.4}",
                                         tested, total_combinations,
                                         100.0 * tested as f64 / total_combinations as f64,
                                         optimize_for,
                                         best_metric);
                            }
                            results.push(result);
                        }
                        Err(e) => {
                            eprintln!("Error evaluating params: {}", e);
                        }
                    }
                }
            }
        }
    }

    // Sort results by optimization metric
    match optimize_for {
        "auc" => results.sort_by(|a, b| b.auc.partial_cmp(&a.auc).unwrap()),
        "log_loss" => results.sort_by(|a, b| a.log_loss.partial_cmp(&b.log_loss).unwrap()),
        "f1_score" => results.sort_by(|a, b| b.f1_score.partial_cmp(&a.f1_score).unwrap()),
        _ => results.sort_by(|a, b| b.auc.partial_cmp(&a.auc).unwrap()),
    }

    println!("\n=== Top 10 Parameter Combinations (Optimized for {}) ===", optimize_for);
    for (i, result) in results.iter().take(10).enumerate() {
        result.print(i + 1);
    }

    Ok(results)
}

fn get_best_metric(results: &[GridSearchResult], optimize_for: &str) -> f64 {
    if results.is_empty() {
        return 0.0;
    }
    
    match optimize_for {
        "auc" => results.iter().map(|r| r.auc).fold(0.0f64, f64::max),
        "log_loss" => results.iter().map(|r| r.log_loss).fold(f64::INFINITY, f64::min),
        "f1_score" => results.iter().map(|r| r.f1_score).fold(0.0f64, f64::max),
        _ => results.iter().map(|r| r.auc).fold(0.0f64, f64::max),
    }
}

async fn evaluate_parameters(
    records: &Vec<FormattedRecord>,
    params: EmResult,
    model: Models,
) -> Result<GridSearchResult, Box<dyn Error>> {
    // Load fresh student data for each parameter set
    let mut users = load_students(records, params).await?;

    // Run evaluation based on model type
    let predictions = match model {
        Models::HiddenMarkovModel => {
            evaluate_hmm(&mut users, records, params.transition, params.slip, params.guess).await
        }
        Models::KnowledgeTracingModel => {
            evaluate_ktm(&mut users, records, params.transition, params.slip, params.guess).await
        }
    };

    // Calculate metrics
    let auc = calculate_auc_roc(predictions.clone());
    let metrics = calculate_classification_metrics(&predictions, 0.5);

    Ok(GridSearchResult {
        initial: params.initial,
        transition: params.transition,
        slip: params.slip,
        guess: params.guess,
        auc,
        log_loss: metrics["log_loss"],
        brier_score: metrics["brier_score"],
        accuracy: metrics["accuracy"],
        f1_score: metrics["f1_score"],
    })
}

