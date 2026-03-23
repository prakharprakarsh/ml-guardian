"""
ML Guardian — Synthetic Data Generator
Generates a credit risk dataset for demonstration purposes.
Simulates realistic Dutch/EU credit data with drift capabilities.
"""
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path


def generate_credit_data(
    n_samples: int = 10000,
    drift: bool = False,
    drift_intensity: float = 0.3,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic credit risk data.
    
    Args:
        n_samples: Number of samples to generate.
        drift: If True, introduce data drift to simulate production shift.
        drift_intensity: How much drift to introduce (0.0 to 1.0).
        seed: Random seed for reproducibility.
    
    Returns:
        DataFrame with credit features and target.
    """
    rng = np.random.RandomState(seed)

    # Base distributions (normal production data)
    age = rng.normal(42, 12, n_samples).clip(18, 75).astype(int)
    income = rng.lognormal(10.5, 0.6, n_samples).clip(15000, 300000)
    employment_years = rng.exponential(8, n_samples).clip(0, 40)
    loan_amount = rng.lognormal(10, 0.8, n_samples).clip(1000, 500000)
    credit_score = rng.normal(680, 80, n_samples).clip(300, 850).astype(int)
    num_credit_lines = rng.poisson(4, n_samples).clip(0, 15)
    debt_to_income = rng.beta(2, 5, n_samples) * 0.8
    months_since_last_delinquency = rng.exponential(24, n_samples).clip(0, 120)
    num_open_accounts = rng.poisson(6, n_samples).clip(1, 20)
    revolving_utilization = rng.beta(2, 5, n_samples)

    # Protected attributes (for fairness monitoring)
    gender = rng.choice(["M", "F", "Other"], n_samples, p=[0.48, 0.48, 0.04])
    age_group = pd.cut(age, bins=[17, 30, 45, 60, 76],
                       labels=["young", "middle", "senior", "elderly"])
    nationality = rng.choice(
        ["NL", "DE", "TR", "MA", "SU", "Other"],
        n_samples,
        p=[0.55, 0.10, 0.10, 0.08, 0.07, 0.10],
    )

    if drift:
        # Simulate realistic drift scenarios
        # Scenario: Economic downturn — incomes shift lower, DTI rises
        income *= (1 - drift_intensity * 0.2)
        debt_to_income += drift_intensity * 0.15
        debt_to_income = debt_to_income.clip(0, 0.95)

        # Scenario: Younger population applying for loans
        age = (age - drift_intensity * 5).clip(18, 75).astype(int)
        age_group = pd.cut(age, bins=[17, 30, 45, 60, 76],
                           labels=["young", "middle", "senior", "elderly"])

        # Scenario: Credit scores shift down
        credit_score = (credit_score - drift_intensity * 30).clip(300, 850).astype(int)

        # Scenario: Higher loan amounts requested
        loan_amount *= (1 + drift_intensity * 0.3)

    # Generate target (default: 0 = no default, 1 = default)
    default_prob = (
        -0.05 * (credit_score - 600) / 100
        + 0.3 * debt_to_income
        - 0.02 * employment_years
        + 0.1 * (loan_amount / income)
        - 0.01 * (age - 40) / 10
        + rng.normal(0, 0.1, n_samples)
    )
    default_prob = 1 / (1 + np.exp(-default_prob))  # sigmoid
    target = (rng.uniform(0, 1, n_samples) < default_prob).astype(int)

    df = pd.DataFrame({
        "age": age,
        "income": np.round(income, 2),
        "employment_years": np.round(employment_years, 1),
        "loan_amount": np.round(loan_amount, 2),
        "credit_score": credit_score,
        "num_credit_lines": num_credit_lines,
        "debt_to_income": np.round(debt_to_income, 4),
        "months_since_delinquency": np.round(months_since_last_delinquency, 0),
        "num_open_accounts": num_open_accounts,
        "revolving_utilization": np.round(revolving_utilization, 4),
        # Protected attributes
        "gender": gender,
        "age_group": age_group.astype(str),
        "nationality": nationality,
        # Target
        "default": target,
    })

    return df


def save_datasets(output_dir: str = "data"):
    """Generate and save reference and production datasets."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Reference data (training distribution)
    ref_data = generate_credit_data(n_samples=10000, drift=False, seed=42)
    ref_data.to_csv(output_path / "reference_data.csv", index=False)

    # Production data — no drift (normal operations)
    prod_normal = generate_credit_data(n_samples=5000, drift=False, seed=123)
    prod_normal.to_csv(output_path / "production_normal.csv", index=False)

    # Production data — with drift (simulated issue)
    prod_drifted = generate_credit_data(
        n_samples=5000, drift=True, drift_intensity=0.4, seed=456
    )
    prod_drifted.to_csv(output_path / "production_drifted.csv", index=False)

    print(f"Datasets saved to {output_path}/")
    print(f"  Reference:  {len(ref_data)} samples (default rate: {ref_data['default'].mean():.2%})")
    print(f"  Prod Normal: {len(prod_normal)} samples (default rate: {prod_normal['default'].mean():.2%})")
    print(f"  Prod Drifted: {len(prod_drifted)} samples (default rate: {prod_drifted['default'].mean():.2%})")


if __name__ == "__main__":
    save_datasets()
