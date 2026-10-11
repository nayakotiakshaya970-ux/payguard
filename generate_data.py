import os
import random

import numpy as np
import pandas as pd
from faker import Faker

fake = Faker()
np.random.seed(42)
random.seed(42)

n = 2000
data = []

for i in range(n):
    client_age_months = np.random.randint(1, 60)
    past_late_payments = np.random.poisson(1.5)
    invoice_amount = round(np.random.uniform(200, 15000), 2)
    payment_terms = random.choice([15, 30, 45, 60])
    industry = random.choice(["Retail", "Tech", "Construction", "Healthcare", "Consulting"])
    client_rating = round(np.random.uniform(1, 5), 1)

    risk_score = (
        (past_late_payments * 0.3)
        + (1 / (client_rating + 0.1) * 0.4)
        + (invoice_amount / 15000 * 0.2)
        + np.random.normal(0, 0.3)
    )
    is_late = 1 if risk_score > 1.0 else 0

    data.append({
        "invoice_id": f"INV-{1000 + i}",
        "client_name": fake.company(),
        "client_age_months": client_age_months,
        "past_late_payments": past_late_payments,
        "invoice_amount": invoice_amount,
        "payment_terms_days": payment_terms,
        "industry": industry,
        "client_rating": client_rating,
        "paid_late": is_late,
    })

df = pd.DataFrame(data)

os.makedirs("data", exist_ok=True)
df.to_csv("data/invoices.csv", index=False)

print(f"Generated {len(df)} invoice records")
print(f"Late payment rate: {df['paid_late'].mean():.1%}")