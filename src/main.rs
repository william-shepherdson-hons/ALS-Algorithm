use als_algorithm::{data::preprocess::process_assistments, evaluation::{em_result::EmResult, expectation_maximisation::expectation_maximisation}};


#[tokio::main]
async fn main(){
    //let _ =  process_assistments();
    let params = EmResult {
        initial: 0.1,
        transition: 0.3,
        slip: 0.1,
        guess: 0.1
    };
    let _ = expectation_maximisation(als_algorithm::models::models::Models::HiddenMarkovModel, params, "src/data/train_data.csv").await;
} 