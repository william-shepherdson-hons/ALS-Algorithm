use serde::{Serialize,Deserialize};
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Iteration {
    pub iteration: usize,
    pub initial: f64,
    pub transition: f64,
    pub slip: f64,
    pub guess: f64,
    pub diff: f64
}