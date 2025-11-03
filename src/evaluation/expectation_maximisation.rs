use crate::evaluation::em_result::EmResult;
use crate::models::models::Models;

pub async fn expectation_maximisation(model: Models) -> EmResult{
    EmResult {
        guess: 0.0,
        transition: 0.0,
        initial: 0.0,
        slip: 0.0
    }
}