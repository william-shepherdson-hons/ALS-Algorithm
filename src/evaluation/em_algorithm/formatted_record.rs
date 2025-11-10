use serde::{Deserialize, Serialize};


#[derive(Debug, Serialize, Deserialize)]
pub struct FormattedRecord {
    pub user_id: u32,
    pub correct: u32,
    pub times_applied: u32,
    pub skill_id: u32,
}