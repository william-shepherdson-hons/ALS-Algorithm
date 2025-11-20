use als_algorithm::{evaluation::grid_search::grid_search::grid_search_hyperparameters, models::models::Models};

#[tokio::main]
async fn main() {
    let results = grid_search_hyperparameters(
        Models::KnowledgeTracingModel,
        "src/data/train_data.csv",
        "auc",
    ).await.unwrap();

    let best = &results[0];
    println!("Best parameters: init={:.3}, trans={:.3}, slip={:.3}, guess={:.3}",
            best.initial, best.transition, best.slip, best.guess);
}