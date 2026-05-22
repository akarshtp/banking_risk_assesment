# Role-Based Access Control (RBAC)

RBAC is enforced during Knowledge Base Retrieval to prevent unauthorized data access (zero leakage).

## Roles Defined
In `config/roles.yaml`:
- **junior_analyst**: Access to 'public' and 'internal' data.
- **senior_underwriter**: Access up to 'restricted' data.
- **credit_head**: Full access to all data.
- **auditor**: Full access for compliance checks.

## Implementation
When the Vector Store is queried, a metadata filter is constructed dynamically based on the active user role, ensuring they only retrieve allowed `confidentiality` tags and document types.
