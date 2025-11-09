pub async fn calculate_mastery(initial: f64, transition: f64) -> f64 {
    initial + transition * (1.0 - initial)
}

pub async fn calculate_success(mastery: f64, slip: f64, guess: f64) -> f64 {
    guess * (1.0 - mastery) + (1.0 - slip) * mastery
}

pub async fn calculate_backward_pair(
    beta_next_k: f64,
    beta_next_u: f64,
    next_observation: bool,
    transition: f64,
    slip: f64,
    guess: f64,
) -> (f64, f64) {
    let b_k = if next_observation { 1.0 - slip } else { slip };
    let b_u = if next_observation { guess } else { 1.0 - guess };
    let a_kk = 1.0;
    let a_uk = transition;
    let a_ku = 0.0;
    let a_uu = 1.0 - transition;
    let beta_k = a_kk * b_k * beta_next_k + a_ku * b_u * beta_next_u;
    let beta_u = a_uk * b_k * beta_next_k + a_uu * b_u * beta_next_u;
    (beta_k, beta_u)
}

pub async fn calculate_transition_expectation_pair(
    alpha_t_k: f64,
    alpha_t_u: f64,
    beta_t1_k: f64,
    beta_t1_u: f64,
    obs_t1: bool,
    transition: f64,
    slip: f64,
    guess: f64,
) -> f64 {
    let b_k = if obs_t1 { 1.0 - slip } else { slip };
    let b_u = if obs_t1 { guess } else { 1.0 - guess };
    let a_uk = transition;
    let a_kk = 1.0;
    let a_ku = 0.0;
    let a_uu = 1.0 - transition;
    let numerator = alpha_t_u * a_uk * b_k * beta_t1_k;
    let term_kk = alpha_t_k * a_kk * b_k * beta_t1_k;
    let term_ku = alpha_t_k * a_ku * b_u * beta_t1_u;
    let term_uk = alpha_t_u * a_uk * b_k * beta_t1_k;
    let term_uu = alpha_t_u * a_uu * b_u * beta_t1_u;
    let denom = term_kk + term_ku + term_uk + term_uu;
    if denom > 0.0 { numerator / denom } else { 0.0 }
}



#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn no_learning() {
        let mastery = calculate_mastery(0.2, 0.0).await;
        assert_eq!(mastery, 0.2);
    }
    #[tokio::test]
    async fn full_learning() {
        let mastery = calculate_mastery(0.2, 1.0).await;
        assert_eq!(mastery, 1.0);
    }
    #[tokio::test]
    async fn parital_learning() {
        let mastery = calculate_mastery(0.2, 0.2).await;
        assert_eq!(mastery, 0.36);
    }
    #[tokio::test]
    async fn continued_learning() {
        let mastery = calculate_mastery(0.2, 0.2).await;
        let mastery2 = calculate_mastery(mastery, 0.2).await;
        assert!((0.2 < mastery) && (mastery < mastery2))
    }
    #[tokio::test]
    async fn slip_test() {
        let correct = calculate_success(1.0, 0.1, 0.6).await;
        assert_eq!(correct, 0.9)
    }
    #[tokio::test]
    async fn guess_test() {
        let correct = calculate_success(0.0, 0.1, 0.5).await;
        assert_eq!(correct, 0.5)
    }
}
