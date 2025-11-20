use als_algorithm::{evaluation::{em_algorithm::{em_result::EmResult, expectation_maximisation::expectation_maximisation}}, models::models::Models};


#[tokio::main]
async fn main(){

    // let results = grid_search_hyperparameters(
    //     Models::KnowledgeTracingModel,
    //     "src/data/train_data.csv",
    //     "auc",
    // ).await.unwrap();

    // // Best parameters
    // let best = &results[0];
    // println!("Best parameters: init={:.3}, trans={:.3}, slip={:.3}, guess={:.3}",
    //         best.initial, best.transition, best.slip, best.guess);



    // let model = Models::KnowledgeTracingModel;
    // let initial = EmResult {
    //     initial: 0.3,
    //     transition: 0.15,
    //     slip: 0.15,
    //     guess: 0.30
    // }; 
    // let input = "src/data/test_data.csv";
    // let _ = benchmark_model_with_auc(model, initial, input).await;






    let model = Models::HiddenMarkovModel;
    let output = "src/data/test.csv";
    let params = EmResult {
        initial: 0.3,
        transition: 0.15,
        slip: 0.15,
        guess: 0.30
    }; 

    let _ = expectation_maximisation(model, params, "src/data/train_data.csv", output).await;

} 