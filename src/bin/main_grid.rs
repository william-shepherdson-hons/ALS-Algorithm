use std::env;

use als_algorithm::{
    evaluation::grid_search::grid_search::grid_search_hyperparameters,
    models::models::Models,
};

#[tokio::main]
async fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        println!(
            "Argument error: expecting 2 got {}",
            args.len() - 1
        );
        println!("Arguments should be: model (hmm or ktm), metric (e.g. auc, accuracy)");
        return;
    }

    // -------- MODEL SELECTION --------
    let model = match args[1].to_uppercase().as_str() {
        "HMM" => Models::HiddenMarkovModel,
        "KTM" => Models::KnowledgeTracingModel,
        _ => {
            println!("Invalid model: please use hmm or ktm");
            return;
        }
    };

    // -------- METRIC SELECTION --------
    let metric = args[2].as_str();

    // -------- RUN GRID SEARCH --------
    let results = match grid_search_hyperparameters(
        model,
        "src/data/train_data.csv",
        metric,
    )
    .await
    {
        Ok(r) => r,
        Err(e) => {
            println!("Grid search failed: {:?}", e);
            return;
        }
    };

    let best = &results[0];

    println!(
        "Best parameters: init={:.3}, trans={:.3}, slip={:.3}, guess={:.3}",
        best.initial, best.transition, best.slip, best.guess
    );
}