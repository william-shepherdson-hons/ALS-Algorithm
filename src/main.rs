use als_algorithm::{data::preprocess::process_assistments, evaluation::em_algorithm::{em_result::EmResult, expectation_maximisation::expectation_maximisation}, models::models::Models};
use als_algorithm::evaluation::performance::performance_benchmark::benchmark_model_performance;

use std::env::{self};
#[tokio::main]
async fn main(){
    let model = Models::KnowledgeTracingModel;
    let initial = EmResult {
        initial: 0.55,
        transition: 0.03,
        slip: 0.15,
        guess: 0.47
    }; 
    let input = "src/data/test_data.csv";
    let _ = benchmark_model_performance(model,initial,input).await;

    // Preprocess Data
    ///////////////////////////////////////////////
    // let _ =  process_assistments();



    //EM algorithm
    ///////////////////////////////////////////////
    // let args : Vec<String> = env::args().collect();
    // if args.len() < 6 {
    //     println!("Arguement error: expecting 5 got {}", args.len() - 1);
    //     println!("Arguements should be: initial, transistion, slip, guess, model (hmm or ktm)");
    //     return
    // }




    // let params = EmResult {
    //     initial: args[1].parse().expect("Failed to parse initial as f64"),
    //     transition: args[2].parse().expect("Failed to parse transition as f64"),
    //     slip: args[3].parse().expect("Failed to parse slip as f64"),
    //     guess: args[4].parse().expect("Failed to parse guess as f64")
    // };
    // let model;
    // let output:&str;
    // if args[5].to_uppercase() == "HMM" {
    //     model = als_algorithm::models::models::Models::HiddenMarkovModel;
    //     output = "src/data/train_on_hmm.csv";

    // }
    // else if args[5].to_uppercase() == "KTM" {
    //     model = als_algorithm::models::models::Models::KnowledgeTracingModel;
    //     output = "src/data/train_on_ktm.csv";
    // }
    // else {
    //     println!("Invalid model: please use hmm or ktm");
    //     return;
    // }

    // let _ = expectation_maximisation(model, params, "src/data/train_data.csv", output).await;

} 