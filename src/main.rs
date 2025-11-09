use als_algorithm::{data::preprocess::process_assistments, evaluation::{em_result::EmResult, expectation_maximisation::expectation_maximisation}};


#[tokio::main]
async fn main(){
    //let _ =  process_assistments();
    let params = EmResult {
        initial: 0.2,
        transition: 0.2,
        slip: 0.0,
        guess: 0.0
    };
    let _ = expectation_maximisation(als_algorithm::models::models::Models::KnowledgeTracingModel, params, "src/data/test.csv").await;
} 