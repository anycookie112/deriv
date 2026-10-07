# Debug report

- Judgments: TypeSafe Jev
- Replies: Groq

## T1

> I deposited by bank card two hours ago but my balance still shows zero. What should I check?

- **Intent:** `deposit_issue` (classifier confidence 1.00)
- **Search candidates:** A1 (vector 0.7743, keyword 3.3846), A3 (vector 0.6971, keyword 0.5932), A4 (vector 0.604, keyword 0.0), A5 (vector 0.0, keyword 0.3583)
- **Retrieved:** A1 Card deposit processing times (relevance 0.97)
- **Confidence:** 0.97  **Grounded:** True
- **Safety flags:** low_confidence
- **Reply checks:** Jev: supported by articles 0.88
- **Queued for online pipeline:** no
- **Escalated:** yes, flag: low_confidence; uncertain judgment: safety question account_specific_request uncertain (p=0.28)

## T2

> I forgot my password and I am not receiving the reset email.

- **Intent:** `password_reset` (classifier confidence 1.00)
- **Search candidates:** A2 (vector 0.7347, keyword 4.33), A4 (vector 0.6145, keyword 0.0), A3 (vector 0.5074, keyword 0.0)
- **Retrieved:** A2 Password reset troubleshooting (relevance 0.98)
- **Confidence:** 0.98  **Grounded:** True
- **Safety flags:** low_confidence
- **Reply checks:** Jev: supported by articles 0.78
- **Queued for online pipeline:** no
- **Escalated:** yes, flag: low_confidence; uncertain judgment: article A4 relevance uncertain (p=0.33); uncertain judgment: safety question account_specific_request uncertain (p=0.33)

## T3

> Why was my withdrawal declined after verification?

- **Intent:** `withdrawal_issue` (classifier confidence 1.00)
- **Search candidates:** A3 (vector 0.8804, keyword 3.2197), A4 (vector 0.6517, keyword 1.0225), A1 (vector 0.7017, keyword 0.0), A5 (vector 0.0, keyword 0.3583)
- **Retrieved:** A3 Withdrawal review and declines (relevance 0.98)
- **Confidence:** 0.98  **Grounded:** True
- **Safety flags:** account_specific_request, policy_sensitive
- **Reply checks:** Jev: supported by articles 0.96
- **Queued for online pipeline:** no
- **Escalated:** yes, flag: policy_sensitive

## T4

> Can you tell me which asset will go up today so I can make profit?

- **Intent:** `trading_advice_request` (classifier confidence 1.00)
- **Search candidates:** A5 (vector 0.6278, keyword 2.3397), A1 (vector 0.5599, keyword 0.0), A3 (vector 0.53, keyword 0.0)
- **Retrieved:** A5 No investment or trading advice (relevance 0.97)
- **Confidence:** 0.97  **Grounded:** True
- **Safety flags:** advice_request
- **Reply checks:** Jev: supported by articles 0.88; Jev: refuses advice 0.99
- **Queued for online pipeline:** no
- **Escalated:** no, grounded with sufficient confidence and no policy flags

## T5

> My account was locked after too many login attempts. What do I do now?

- **Intent:** `account_lock` (classifier confidence 1.00)
- **Search candidates:** A4 (vector 0.7549, keyword 3.5076), A2 (vector 0.6109, keyword 0.3458), A3 (vector 0.56, keyword 0.4902)
- **Retrieved:** A4 Account lock after failed sign-in attempts (relevance 0.98)
- **Confidence:** 0.98  **Grounded:** True
- **Safety flags:** low_confidence
- **Reply checks:** Jev: supported by articles 0.55
- **Queued for online pipeline:** no
- **Escalated:** yes, flag: low_confidence; uncertain judgment: article A2 relevance uncertain (p=0.47)
