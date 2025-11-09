use als_algorithm::{data::preprocess::process_assistments, evaluation::{em_result::EmResult, expectation_maximisation::expectation_maximisation}};


#[tokio::main]
async fn main(){
    //let _ =  process_assistments();
    let params = EmResult {
        initial: 0.5,
        transition: 0.5,
        slip: 0.3,
        guess: 0.3
    };
    let _ = expectation_maximisation(als_algorithm::models::models::Models::KnowledgeTracingModel, params, "src/data/test.csv").await;
} 