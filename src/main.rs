use als_algorithm::{
    evaluation::em_algorithm::{
        em_result::EmResult, expectation_maximisation::expectation_maximisation,
    },
    models::models::Models,
};

#[tokio::main]
async fn main() {
    let model = Models::HiddenMarkovModel;
    let output = "src/data/test.csv";
    let params = EmResult {
        initial: 0.3,
        transition: 0.15,
        slip: 0.15,
        guess: 0.30,
    };

    let _ = expectation_maximisation(model, params, "src/data/train_data.csv", output).await;
}
