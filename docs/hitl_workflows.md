# HITL Workflows

Human-In-The-Loop (HITL) workflows ensure high-risk decisions are intercepted for manual approval.

## Trigger Rules
Defined in `config/hitl_rules.yaml`:
- **high_value_loan**: Loans > ₹50L (Requires Senior Underwriter).
- **high_dti_ratio**: DTI > 50% (Requires Senior Underwriter).
- **low_credit_score**: Credit Score < 600 (Requires Credit Head).

## Process
When a trigger is hit, the API returns a paused state with a `Task ID`. The pending task is visible in the Streamlit UI's "HITL Approvals" tab for resolution.
