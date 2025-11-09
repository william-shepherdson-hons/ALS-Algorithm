pub async fn calculate_mastery(
    initial: f64,
    transition: f64,
    slip: f64,
    guess: f64,
    correct: bool,
) -> f64 {
    if slip + guess > 1.0 {
        panic!("Invalid parameters: P(G) + P(S) > 1")
    }
    if correct {
        let top = (1.0 - transition) * (1.0 - initial) * guess;
        let bottom = guess + (1.0 - slip - guess) * initial;
        return 1.0 - (top / bottom);
    }
    let top = (1.0 - transition) * (1.0 - initial) * (1.0 - guess);
    let bottom = 1.0 - guess - (1.0 - slip - guess) * initial;
    1.0 - (top / bottom)
}

pub async fn calculate_success(mastery: f64, slip: f64, guess: f64) -> f64 {
    guess * (1.0 - mastery) + (1.0 - slip) * mastery
}

pub async fn calculate_backward_pair_kt(
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

pub async fn calculate_transition_expectation_kt(
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
    async fn typical_answer_correct() {
        let mastery = calculate_mastery(0.3, 0.2, 0.1, 0.1, true).await;
        assert!((mastery - 0.835).abs() < 0.001);
    }

    #[tokio::test]
    async fn typical_answer_incorrect() {
        let mastery = calculate_mastery(0.3, 0.2, 0.1, 0.1, false).await;
        assert!((mastery - 0.236).abs() < 0.001);
    }

    #[tokio::test]
    async fn high_guess_answer_correct() {
        let mastery = calculate_mastery(0.3, 0.2, 0.1, 0.9, true).await;
        assert!((mastery - 0.440).abs() < 0.001);
    }

    #[tokio::test]
    async fn high_guess_answer_incorrect() {
        let mastery = calculate_mastery(0.3, 0.2, 0.1, 0.9, false).await;
        assert!((mastery - 0.440).abs() < 0.001);
    }

    #[tokio::test]
    async fn high_learning_answer_correct() {
        let mastery = calculate_mastery(0.8, 0.9, 0.1, 0.1, true).await;
        assert!((mastery - 0.997).abs() < 0.001);
    }

    #[tokio::test]
    async fn high_learning_answer_incorrect() {
        let mastery = calculate_mastery(0.8, 0.9, 0.1, 0.1, false).await;
        assert!((mastery - 0.931).abs() < 0.001);
    }

    #[tokio::test]
    async fn no_guess_answer_correct() {
        let mastery = calculate_mastery(0.5, 0.5, 0.1, 0.0, true).await;
        assert!((mastery - 1.0).abs() < 0.001);
    }

    #[tokio::test]
    async fn no_guess_answer_incorrect() {
        let mastery = calculate_mastery(0.5, 0.5, 0.1, 0.0, false).await;
        assert!((mastery - 0.545).abs() < 0.001);
    }

    #[tokio::test]
    async fn low_prior_knowledge_correct() {
        let mastery = calculate_mastery(0.1, 0.1, 0.2, 0.4, true).await;
        assert!((mastery - 0.264).abs() < 0.001);
    }

    #[tokio::test]
    async fn low_prior_knowledge_incorrect() {
        let mastery = calculate_mastery(0.1, 0.1, 0.2, 0.4, false).await;
        assert!((mastery - 0.132).abs() < 0.001);
    }

    #[tokio::test]
    async fn mastered_correct() {
        let mastery = calculate_mastery(1.0, 0.4, 0.1, 0.3, true).await;
        assert!((mastery - 1.0).abs() < 0.001);
    }

    #[tokio::test]
    async fn mastered_incorrect() {
        let mastery = calculate_mastery(1.0, 0.4, 0.1, 0.3, false).await;
        assert!((mastery - 1.0).abs() < 0.001);
    }

    #[tokio::test]
    async fn ignorance_correct() {
        let mastery = calculate_mastery(0.0, 0.4, 0.1, 0.3, true).await;
        assert!((mastery - 0.4).abs() < 0.001);
    }

    #[tokio::test]
    async fn ignorance_incorrect() {
        let mastery = calculate_mastery(0.0, 0.4, 0.1, 0.3, false).await;
        assert!((mastery - 0.4).abs() < 0.001);
    }

    #[tokio::test]
    #[should_panic(expected = "Invalid parameters: P(G) + P(S) > 1")]
    async fn no_negative_learning_correct() {
        calculate_mastery(0.3, 0.2, 0.8, 0.4, true).await;
    }

    #[tokio::test]
    #[should_panic(expected = "Invalid parameters: P(G) + P(S) > 1")]
    async fn no_negative_learning_incorrect() {
        calculate_mastery(0.3, 0.2, 0.8, 0.4, false).await;
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
