use als_algorithm::{data::preprocess::process_assistments, evaluation::em_algorithm::{em_result::EmResult, expectation_maximisation::expectation_maximisation}};


#[tokio::main]
async fn main(){
    //let _ =  process_assistments();
    let params = EmResult {
        initial: 0.4,
        transition: 0.1,
        slip: 0.0,
        guess: 0.0
    };
    let _ = expectation_maximisation(als_algorithm::models::models::Models::HiddenMarkovModel, params, "src/data/train_data.csv", "src/data/train_on_hmm.csv").await;
} 