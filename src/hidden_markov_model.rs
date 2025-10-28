pub fn calculate_mastery(initial: f32, transition: f32) -> f32  {
    0.1
}
pub fn calculate_success(mastery: f32, slip: f32, guess: f32) -> f32 {
    0.1
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_learning() {
        let mastery = calculate_mastery(0.2, 0.0);
        assert_eq!(mastery, 0.2);
    }
    #[test]
    fn full_learning() {
        let mastery = calculate_mastery(0.2, 1.0);
        assert_eq!(mastery, 0.2);
    }
    #[test]
    fn parital_learning() {
        let mastery = calculate_mastery(0.2, 0.2);
        assert_eq!(mastery, 0.36);
    }
    #[test]
    fn continued_learning() {
        let mastery = calculate_mastery(0.2, 0.2);
        let mastery2 = calculate_mastery(mastery, 0.2);
        assert!((0.2 < mastery) & (mastery < mastery2))
    }
    #[test]
    fn slip_test() {
        let correct = calculate_success(1.0, 0.1, 0.6);
        assert_eq!(correct, 0.9)
    }
    #[test]
    fn guess_test() {
        let correct = calculate_success(0.0, 0.1, 0.5);
        assert_eq!(correct, 0.5)
    }
}
