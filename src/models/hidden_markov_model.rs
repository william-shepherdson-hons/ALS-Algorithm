pub async fn calculate_mastery(initial: f64, transition: f64) -> f64  {
    let mastery = initial + transition * (1.0 - initial);
    mastery
}
pub async fn calculate_success(mastery: f64, slip: f64, guess: f64) -> f64 {
    let sucess = guess * (1.0 - mastery) + (1.0 - slip) * mastery;
    sucess
}
pub async fn calculate_backward_probability(current_backward: f64, next_observation: bool, slip: f64 , guess: f64, transition: f64) -> f64 {
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
    

    0.1
}

pub async fn calculate_transistion_expectation(forward_prob: f64, backward_prob: f64, next_backward: f64, next_observation: bool, transition: f64, slip: f64, guess: f64) -> f64 {
    0.1
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
