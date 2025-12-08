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
    let initial_range = vec![0.05, 0.10, 0.15, 0.20, 0.25, 0.30];
    let transition_range = vec![0.05, 0.10, 0.15, 0.20, 0.25, 0.30];
    let slip_range = vec![0.05, 0.10, 0.15, 0.20];
    let guess_range = vec![0.10, 0.15, 0.20, 0.25, 0.30];

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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    #[test]
    fn test_grid_search_result_creation() {
        let result = GridSearchResult {
            initial: 0.3,
            transition: 0.2,
            slip: 0.1,
            guess: 0.25,
            auc: 0.85,
            log_loss: 0.35,
            brier_score: 0.15,
            accuracy: 0.82,
            f1_score: 0.78,
        };

        assert_relative_eq!(result.initial, 0.3, epsilon = 1e-10);
        assert_relative_eq!(result.transition, 0.2, epsilon = 1e-10);
        assert_relative_eq!(result.slip, 0.1, epsilon = 1e-10);
        assert_relative_eq!(result.guess, 0.25, epsilon = 1e-10);
        assert_relative_eq!(result.auc, 0.85, epsilon = 1e-10);
        assert_relative_eq!(result.log_loss, 0.35, epsilon = 1e-10);
        assert_relative_eq!(result.brier_score, 0.15, epsilon = 1e-10);
        assert_relative_eq!(result.accuracy, 0.82, epsilon = 1e-10);
        assert_relative_eq!(result.f1_score, 0.78, epsilon = 1e-10);
    }

    #[test]
    fn test_grid_search_result_print() {
        let result = GridSearchResult {
            initial: 0.3,
            transition: 0.2,
            slip: 0.1,
            guess: 0.25,
            auc: 0.85,
            log_loss: 0.35,
            brier_score: 0.15,
            accuracy: 0.82,
            f1_score: 0.78,
        };

        // Test that print doesn't panic
        result.print(1);
    }

    #[test]
    fn test_get_best_metric_empty_results() {
        let results: Vec<GridSearchResult> = vec![];
        
        // get_best_metric checks if results.is_empty() first and returns 0.0
        assert_relative_eq!(get_best_metric(&results, "auc"), 0.0, epsilon = 1e-10);
        assert_relative_eq!(get_best_metric(&results, "f1_score"), 0.0, epsilon = 1e-10);
        assert_relative_eq!(get_best_metric(&results, "log_loss"), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_get_best_metric_auc() {
        let results = vec![
            GridSearchResult {
                initial: 0.3,
                transition: 0.2,
                slip: 0.1,
                guess: 0.25,
                auc: 0.75,
                log_loss: 0.4,
                brier_score: 0.2,
                accuracy: 0.70,
                f1_score: 0.65,
            },
            GridSearchResult {
                initial: 0.25,
                transition: 0.15,
                slip: 0.15,
                guess: 0.2,
                auc: 0.85,
                log_loss: 0.35,
                brier_score: 0.18,
                accuracy: 0.82,
                f1_score: 0.78,
            },
            GridSearchResult {
                initial: 0.2,
                transition: 0.25,
                slip: 0.12,
                guess: 0.22,
                auc: 0.80,
                log_loss: 0.38,
                brier_score: 0.19,
                accuracy: 0.75,
                f1_score: 0.72,
            },
        ];

        let best_auc = get_best_metric(&results, "auc");
        assert_relative_eq!(best_auc, 0.85, epsilon = 1e-10);
    }

    #[test]
    fn test_get_best_metric_log_loss() {
        let results = vec![
            GridSearchResult {
                initial: 0.3,
                transition: 0.2,
                slip: 0.1,
                guess: 0.25,
                auc: 0.75,
                log_loss: 0.4,
                brier_score: 0.2,
                accuracy: 0.70,
                f1_score: 0.65,
            },
            GridSearchResult {
                initial: 0.25,
                transition: 0.15,
                slip: 0.15,
                guess: 0.2,
                auc: 0.85,
                log_loss: 0.25,
                brier_score: 0.18,
                accuracy: 0.82,
                f1_score: 0.78,
            },
            GridSearchResult {
                initial: 0.2,
                transition: 0.25,
                slip: 0.12,
                guess: 0.22,
                auc: 0.80,
                log_loss: 0.38,
                brier_score: 0.19,
                accuracy: 0.75,
                f1_score: 0.72,
            },
        ];

        let best_log_loss = get_best_metric(&results, "log_loss");
        assert_relative_eq!(best_log_loss, 0.25, epsilon = 1e-10);
    }

    #[test]
    fn test_get_best_metric_f1_score() {
        let results = vec![
            GridSearchResult {
                initial: 0.3,
                transition: 0.2,
                slip: 0.1,
                guess: 0.25,
                auc: 0.75,
                log_loss: 0.4,
                brier_score: 0.2,
                accuracy: 0.70,
                f1_score: 0.65,
            },
            GridSearchResult {
                initial: 0.25,
                transition: 0.15,
                slip: 0.15,
                guess: 0.2,
                auc: 0.85,
                log_loss: 0.35,
                brier_score: 0.18,
                accuracy: 0.82,
                f1_score: 0.82,
            },
            GridSearchResult {
                initial: 0.2,
                transition: 0.25,
                slip: 0.12,
                guess: 0.22,
                auc: 0.80,
                log_loss: 0.38,
                brier_score: 0.19,
                accuracy: 0.75,
                f1_score: 0.72,
            },
        ];

        let best_f1 = get_best_metric(&results, "f1_score");
        assert_relative_eq!(best_f1, 0.82, epsilon = 1e-10);
    }

    #[test]
    fn test_get_best_metric_unknown_metric_defaults_to_auc() {
        let results = vec![
            GridSearchResult {
                initial: 0.3,
                transition: 0.2,
                slip: 0.1,
                guess: 0.25,
                auc: 0.90,
                log_loss: 0.4,
                brier_score: 0.2,
                accuracy: 0.70,
                f1_score: 0.65,
            },
        ];

        let best = get_best_metric(&results, "unknown_metric");
        assert_relative_eq!(best, 0.90, epsilon = 1e-10);
    }

    #[test]
    fn test_parameter_validation_slip_plus_guess() {
        // Test that slip + guess > 1.0 combinations would be skipped
        let slip = 0.6;
        let guess = 0.5;
        
        assert!(slip + guess > 1.0, "This combination should be invalid");
    }

    #[test]
    fn test_valid_parameter_combination() {
        let slip = 0.1;
        let guess = 0.25;
        
        assert!(slip + guess <= 1.0, "This combination should be valid");
    }

    #[test]
    fn test_grid_search_result_sorting_by_auc() {
        let mut results = vec![
            GridSearchResult {
                initial: 0.3,
                transition: 0.2,
                slip: 0.1,
                guess: 0.25,
                auc: 0.75,
                log_loss: 0.4,
                brier_score: 0.2,
                accuracy: 0.70,
                f1_score: 0.65,
            },
            GridSearchResult {
                initial: 0.25,
                transition: 0.15,
                slip: 0.15,
                guess: 0.2,
                auc: 0.85,
                log_loss: 0.35,
                brier_score: 0.18,
                accuracy: 0.82,
                f1_score: 0.78,
            },
            GridSearchResult {
                initial: 0.2,
                transition: 0.25,
                slip: 0.12,
                guess: 0.22,
                auc: 0.80,
                log_loss: 0.38,
                brier_score: 0.19,
                accuracy: 0.75,
                f1_score: 0.72,
            },
        ];

        results.sort_by(|a, b| b.auc.partial_cmp(&a.auc).unwrap());

        assert_relative_eq!(results[0].auc, 0.85, epsilon = 1e-10);
        assert_relative_eq!(results[1].auc, 0.80, epsilon = 1e-10);
        assert_relative_eq!(results[2].auc, 0.75, epsilon = 1e-10);
    }

    #[test]
    fn test_grid_search_result_sorting_by_log_loss() {
        let mut results = vec![
            GridSearchResult {
                initial: 0.3,
                transition: 0.2,
                slip: 0.1,
                guess: 0.25,
                auc: 0.75,
                log_loss: 0.4,
                brier_score: 0.2,
                accuracy: 0.70,
                f1_score: 0.65,
            },
            GridSearchResult {
                initial: 0.25,
                transition: 0.15,
                slip: 0.15,
                guess: 0.2,
                auc: 0.85,
                log_loss: 0.25,
                brier_score: 0.18,
                accuracy: 0.82,
                f1_score: 0.78,
            },
            GridSearchResult {
                initial: 0.2,
                transition: 0.25,
                slip: 0.12,
                guess: 0.22,
                auc: 0.80,
                log_loss: 0.38,
                brier_score: 0.19,
                accuracy: 0.75,
                f1_score: 0.72,
            },
        ];

        results.sort_by(|a, b| a.log_loss.partial_cmp(&b.log_loss).unwrap());

        assert_relative_eq!(results[0].log_loss, 0.25, epsilon = 1e-10);
        assert_relative_eq!(results[1].log_loss, 0.38, epsilon = 1e-10);
        assert_relative_eq!(results[2].log_loss, 0.4, epsilon = 1e-10);
    }

    #[test]
    fn test_grid_search_result_sorting_by_f1_score() {
        let mut results = vec![
            GridSearchResult {
                initial: 0.3,
                transition: 0.2,
                slip: 0.1,
                guess: 0.25,
                auc: 0.75,
                log_loss: 0.4,
                brier_score: 0.2,
                accuracy: 0.70,
                f1_score: 0.65,
            },
            GridSearchResult {
                initial: 0.25,
                transition: 0.15,
                slip: 0.15,
                guess: 0.2,
                auc: 0.85,
                log_loss: 0.35,
                brier_score: 0.18,
                accuracy: 0.82,
                f1_score: 0.82,
            },
            GridSearchResult {
                initial: 0.2,
                transition: 0.25,
                slip: 0.12,
                guess: 0.22,
                auc: 0.80,
                log_loss: 0.38,
                brier_score: 0.19,
                accuracy: 0.75,
                f1_score: 0.72,
            },
        ];

        results.sort_by(|a, b| b.f1_score.partial_cmp(&a.f1_score).unwrap());

        assert_relative_eq!(results[0].f1_score, 0.82, epsilon = 1e-10);
        assert_relative_eq!(results[1].f1_score, 0.72, epsilon = 1e-10);
        assert_relative_eq!(results[2].f1_score, 0.65, epsilon = 1e-10);
    }

    #[test]
    fn test_grid_search_result_clone() {
        let result = GridSearchResult {
            initial: 0.3,
            transition: 0.2,
            slip: 0.1,
            guess: 0.25,
            auc: 0.85,
            log_loss: 0.35,
            brier_score: 0.15,
            accuracy: 0.82,
            f1_score: 0.78,
        };

        let cloned = result.clone();

        assert_relative_eq!(cloned.initial, result.initial, epsilon = 1e-10);
        assert_relative_eq!(cloned.transition, result.transition, epsilon = 1e-10);
        assert_relative_eq!(cloned.slip, result.slip, epsilon = 1e-10);
        assert_relative_eq!(cloned.guess, result.guess, epsilon = 1e-10);
        assert_relative_eq!(cloned.auc, result.auc, epsilon = 1e-10);
        assert_relative_eq!(cloned.log_loss, result.log_loss, epsilon = 1e-10);
        assert_relative_eq!(cloned.brier_score, result.brier_score, epsilon = 1e-10);
        assert_relative_eq!(cloned.accuracy, result.accuracy, epsilon = 1e-10);
        assert_relative_eq!(cloned.f1_score, result.f1_score, epsilon = 1e-10);
    }

    #[test]
    fn test_grid_dimension_calculations() {
        let initial_range = vec![0.05, 0.10, 0.15, 0.20, 0.25, 0.30];
        let transition_range = vec![0.05, 0.10, 0.15, 0.20, 0.25, 0.30];
        let slip_range = vec![0.05, 0.10, 0.15, 0.20];
        let guess_range = vec![0.10, 0.15, 0.20, 0.25, 0.30];

        let total_combinations = initial_range.len() 
            * transition_range.len() 
            * slip_range.len() 
            * guess_range.len();
        
        assert_eq!(total_combinations, 6 * 6 * 4 * 5);
        assert_eq!(total_combinations, 720);
    }

    #[test]
    fn test_valid_combinations_filtering() {
        // Count how many valid combinations exist in a small grid
        let slip_range = vec![0.1, 0.3, 0.5];
        let guess_range = vec![0.2, 0.4, 0.6];
        
        let valid_count = slip_range.iter()
            .flat_map(|&s| guess_range.iter().map(move |&g| (s, g)))
            .filter(|&(s, g)| s + g <= 1.0)
            .count();
        assert_eq!(valid_count, 8);
    }

    #[test]
    fn test_parameter_ranges_are_valid() {
        let initial_range = vec![0.05, 0.10, 0.15, 0.20, 0.25, 0.30];
        let transition_range = vec![0.05, 0.10, 0.15, 0.20, 0.25, 0.30];
        let slip_range = vec![0.05, 0.10, 0.15, 0.20];
        let guess_range = vec![0.10, 0.15, 0.20, 0.25, 0.30];

        // All values should be in valid probability range [0, 1]
        for &val in &initial_range {
            assert!(val >= 0.0 && val <= 1.0);
        }
        for &val in &transition_range {
            assert!(val >= 0.0 && val <= 1.0);
        }
        for &val in &slip_range {
            assert!(val >= 0.0 && val <= 1.0);
        }
        for &val in &guess_range {
            assert!(val >= 0.0 && val <= 1.0);
        }
    }

    #[test]
    fn test_metric_comparison_functions() {
        let a = GridSearchResult {
            initial: 0.3,
            transition: 0.2,
            slip: 0.1,
            guess: 0.25,
            auc: 0.85,
            log_loss: 0.35,
            brier_score: 0.15,
            accuracy: 0.82,
            f1_score: 0.78,
        };
        
        let b = GridSearchResult {
            initial: 0.25,
            transition: 0.15,
            slip: 0.15,
            guess: 0.2,
            auc: 0.80,
            log_loss: 0.30,
            brier_score: 0.18,
            accuracy: 0.75,
            f1_score: 0.72,
        };

        // AUC: higher is better (a > b)
        assert!(a.auc > b.auc);
        assert!(b.auc.partial_cmp(&a.auc).unwrap() == std::cmp::Ordering::Less);

        // Log loss: lower is better (b < a)
        assert!(b.log_loss < a.log_loss);
        assert!(a.log_loss.partial_cmp(&b.log_loss).unwrap() == std::cmp::Ordering::Greater);

        // F1: higher is better (a > b)
        assert!(a.f1_score > b.f1_score);
        assert!(b.f1_score.partial_cmp(&a.f1_score).unwrap() == std::cmp::Ordering::Less);
    }

    #[test]
    fn test_results_take_10() {
        let results: Vec<GridSearchResult> = (0..20)
            .map(|i| GridSearchResult {
                initial: 0.3,
                transition: 0.2,
                slip: 0.1,
                guess: 0.25,
                auc: 0.5 + (i as f64 * 0.02),
                log_loss: 0.5,
                brier_score: 0.2,
                accuracy: 0.7,
                f1_score: 0.6,
            })
            .collect();

        let top_10: Vec<_> = results.iter().take(10).collect();
        assert_eq!(top_10.len(), 10);
    }

    #[test]
    fn test_results_take_10_with_fewer_results() {
        let results: Vec<GridSearchResult> = (0..5)
            .map(|i| GridSearchResult {
                initial: 0.3,
                transition: 0.2,
                slip: 0.1,
                guess: 0.25,
                auc: 0.5 + (i as f64 * 0.02),
                log_loss: 0.5,
                brier_score: 0.2,
                accuracy: 0.7,
                f1_score: 0.6,
            })
            .collect();

        let top_10: Vec<_> = results.iter().take(10).collect();
        assert_eq!(top_10.len(), 5);
    }
}