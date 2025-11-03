pub async fn calculate_mastery(initial: f32, transition: f32) -> f32  {
    let mastery = initial + transition * (1.0 - initial);
    mastery
}
pub async fn calculate_success(mastery: f32, slip: f32, guess: f32) -> f32 {
    let sucess = guess * (1.0 - mastery) + (1.0 - slip) * mastery;
    sucess
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
