pub async fn calculate_mastery(initial: f64, transition: f64, slip: f64, guess: f64, correct: bool) -> f64  {
    if slip + guess > 1.0 {
        panic!("Invalid parameters: P(G) + P(S) > 1")
    }
    if correct {
        let top = (1.0 - transition) * (1.0 - initial) * guess;
        let bottom = guess + (1.0 - slip - guess) * initial;
        let mastery = 1.0 - (top/bottom);
        return mastery;
    }
    let top = (1.0 - transition) * (1.0 - initial) * (1.0 - guess);
    let bottom = 1.0 - guess - (1.0 - slip - guess) * initial;
    let mastery = 1.0 - (top/bottom);
    mastery
}
pub async fn calculate_success(mastery: f64, slip: f64, guess: f64) -> f64 {
    let sucess = guess * (1.0 - mastery) + (1.0 - slip) * mastery;
    sucess
}

pub async fn calculate_backward_probability(current_backward: f64, next_observation: bool, slip: f64, guess: f64) -> f64 {
    let p_obs_if_known = if next_observation {
        1.0 - slip
    } else {
        slip
    };
    
    let p_obs_if_unknown = if next_observation {
        guess
    } else {
        1.0 - guess
    };
    
    let beta_if_known = p_obs_if_known * current_backward;
    let beta_if_unknown = p_obs_if_unknown * current_backward;
    
    beta_if_known.max(beta_if_unknown)
}

pub async fn calculate_transition_expectation(forward_prob: f64, next_forward_prob: f64) -> f64 {
    let p_unknown_t = 1.0 - forward_prob;
    
    let knowledge_increase = (next_forward_prob - forward_prob).max(0.0);
    let xi = p_unknown_t * knowledge_increase;
    
    xi
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
