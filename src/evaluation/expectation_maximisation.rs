use crate::evaluation::em_result::EmResult;

pub async fn expectation_maximisation() -> EmResult{
    EmResult {
        guess: 0.0,
        transition: 0.0,
        initial: 0.0,
        slip: 0.0
    }
}