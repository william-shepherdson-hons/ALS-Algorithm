use std::env;

use als_algorithm::evaluation::{em_algorithm::em_result::EmResult, performance::auc_roc::benchmark_model_with_auc};

#[tokio::main]
async fn main() {
    let args : Vec<String> = env::args().collect();
    if args.len() < 6 {
        println!("Arguement error: expecting 5 got {}", args.len() - 1);
        println!("Arguements should be: initial, transistion, slip, guess, model (hmm or ktm)");
        return
    }

    let params = EmResult {
        initial: args[1].parse().expect("Failed to parse initial as f64"),
        transition: args[2].parse().expect("Failed to parse transition as f64"),
        slip: args[3].parse().expect("Failed to parse slip as f64"),
        guess: args[4].parse().expect("Failed to parse guess as f64")
    };
    let model;
    if args[5].to_uppercase() == "HMM" {
        model = als_algorithm::models::models::Models::HiddenMarkovModel;

    }
    else if args[5].to_uppercase() == "KTM" {
        model = als_algorithm::models::models::Models::KnowledgeTracingModel;
    }
    else {
        println!("Invalid model: please use hmm or ktm");
        return;
    }
    let input = "src/data/test_data.csv";
    let _ = benchmark_model_with_auc(model, params, input).await;
}