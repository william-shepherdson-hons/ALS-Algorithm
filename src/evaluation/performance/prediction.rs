#[derive(Debug, Clone)]
pub struct Prediction {
    pub probability: f64,
    pub actual: bool,
}