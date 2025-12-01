import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_loan_dataset(n_samples=50000):
    """
    Generate a comprehensive dataset for loan eligibility prediction
    """
    
    # Initialize lists to store data
    data = []
    
    for i in range(n_samples):
        # Basic demographics
        age = np.random.randint(22, 70)
        
        # Income distribution - more realistic with some high earners
        income = np.random.lognormal(10.8, 0.6)  # Mean around ~60k
        income = max(20000, min(income, 300000))  # Cap between 20k-300k
        
        # Education level with realistic distribution
        education = np.random.choice(
            ['High School', 'Some College', 'Bachelor', 'Master', 'PhD'],
            p=[0.25, 0.25, 0.3, 0.15, 0.05]
        )
        
        # Employment years - correlated with age
        employment_years = max(0, min(age - 22, np.random.exponential(8)))
        
        # Credit score - normally distributed around 650
        credit_score = int(np.random.normal(650, 100))
        credit_score = max(300, min(credit_score, 850))
        
        # Debt-to-income ratio
        debt_to_income = np.random.beta(2, 5) * 0.8 + 0.1  # Mostly low values
        
        # Existing loans
        existing_loans = np.random.poisson(1.5)
        existing_loans = min(existing_loans, 6)
        
        # Savings balance - correlated with income
        savings_balance = np.random.exponential(income * 0.3)
        
        # Mortgage balance - only for some customers
        has_mortgage = np.random.random() < 0.6
        if has_mortgage:
            mortgage_balance = np.random.exponential(150000)
            mortgage_balance = min(mortgage_balance, 1000000)
        else:
            mortgage_balance = 0
        
        # Loan amount requested - correlated with income
        loan_amount_requested = np.random.exponential(income * 0.3)
        loan_amount_requested = max(1000, min(loan_amount_requested, 100000))
        
        # Additional features
        years_at_current_address = np.random.exponential(5)
        number_of_credit_cards = np.random.poisson(2.5)
        late_payments_6m = np.random.poisson(0.5)
        bankruptcies = 1 if np.random.random() < 0.05 else 0
        
        # Marital status
        marital_status = np.random.choice(
            ['Single', 'Married', 'Divorced', 'Widowed'],
            p=[0.4, 0.45, 0.1, 0.05]
        )
        
        # Dependents
        dependents = np.random.poisson(0.8)
        dependents = min(dependents, 5)
        
        # Home ownership
        home_ownership = np.random.choice(
            ['Rent', 'Own', 'Mortgage', 'With Parents'],
            p=[0.3, 0.2, 0.4, 0.1]
        )
        
        # Create target variable based on realistic business rules
        # Higher probability of approval for:
        # - Higher income
        # - Better credit score
        # - Lower DTI
        # - Fewer existing loans
        # - Reasonable loan amount
        
        base_approval_prob = 0.3  # Base approval rate
        
        # Adjust probability based on features
        income_factor = min(1.0, income / 80000)  # Normalize income effect
        credit_factor = max(0, min(1.0, (credit_score - 500) / 350))  # 500-850 scale
        dti_factor = max(0, 1 - debt_to_income)  # Lower DTI is better
        loan_factor = min(1.0, (income * 0.5) / loan_amount_requested)  # Loan shouldn't be too high relative to income
        employment_factor = min(1.0, employment_years / 10)  # More employment years is better
        
        # Combine factors
        approval_prob = (
            base_approval_prob +
            income_factor * 0.2 +
            credit_factor * 0.2 +
            dti_factor * 0.15 +
            loan_factor * 0.1 +
            employment_factor * 0.05
        ) / 1.7  # Normalize to reasonable range
        
        # Additional penalties
        if late_payments_6m > 2:
            approval_prob *= 0.7
        if bankruptcies > 0:
            approval_prob *= 0.5
        if existing_loans > 3:
            approval_prob *= 0.8
        
        # Ensure probability is between 0 and 1
        approval_prob = max(0.05, min(0.95, approval_prob))
        
        # Determine final approval
        loan_approved = 1 if np.random.random() < approval_prob else 0
        
        # Create row
        row = {
            'customer_id': f'CUST_{i+1:06d}',
            'age': age,
            'income': round(income, 2),
            'education': education,
            'employment_years': round(employment_years, 1),
            'credit_score': credit_score,
            'debt_to_income_ratio': round(debt_to_income, 3),
            'existing_loans': existing_loans,
            'savings_balance': round(savings_balance, 2),
            'mortgage_balance': round(mortgage_balance, 2),
            'loan_amount_requested': round(loan_amount_requested, 2),
            'years_at_current_address': round(years_at_current_address, 1),
            'number_of_credit_cards': number_of_credit_cards,
            'late_payments_6m': late_payments_6m,
            'bankruptcies': bankruptcies,
            'marital_status': marital_status,
            'dependents': dependents,
            'home_ownership': home_ownership,
            'loan_approved': loan_approved,
            'approval_probability': round(approval_prob, 3)
        }
        
        data.append(row)
    
    return pd.DataFrame(data)

# Generate the dataset
print("Generating 50,000 row dataset...")
df = generate_loan_dataset(50000)

# Display dataset info
print("Dataset generated successfully!")
print(f"Dataset shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())

print("\nDataset info:")
print(df.info())

print("\nTarget variable distribution:")
print(df['loan_approved'].value_counts())
print(f"Approval rate: {df['loan_approved'].mean():.2%}")

print("\nBasic statistics:")
print(df.describe())

# Save to CSV
csv_filename = 'loan_eligibility_dataset_50k.csv'
df.to_csv(csv_filename, index=False)
print(f"\nDataset saved as '{csv_filename}'")

# Create a sample of the data for quick inspection
sample_df = df.sample(1000)  # 1000 random samples
sample_filename = 'loan_eligibility_sample_1k.csv'
sample_df.to_csv(sample_filename, index=False)
print(f"Sample dataset saved as '{sample_filename}'")

# Additional analysis
print("\n=== Additional Analysis ===")
print("\nEducation level distribution:")
print(df['education'].value_counts())

print("\nHome ownership distribution:")
print(df['home_ownership'].value_counts())

print("\nCorrelation with target variable:")
# Calculate correlation for numerical features
numerical_features = ['age', 'income', 'employment_years', 'credit_score', 
                     'debt_to_income_ratio', 'existing_loans', 'savings_balance',
                     'mortgage_balance', 'loan_amount_requested']

correlations = df[numerical_features + ['loan_approved']].corr()['loan_approved'].sort_values(ascending=False)
print(correlations)

# Create some visualizations (optional - requires matplotlib)
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Income distribution by loan approval
    sns.boxplot(data=df, x='loan_approved', y='income', ax=axes[0, 0])
    axes[0, 0].set_title('Income Distribution by Loan Approval')
    axes[0, 0].set_xlabel('Loan Approved (0=No, 1=Yes)')
    axes[0, 0].set_ylabel('Income')
    
    # Plot 2: Credit score distribution by loan approval
    sns.boxplot(data=df, x='loan_approved', y='credit_score', ax=axes[0, 1])
    axes[0, 1].set_title('Credit Score Distribution by Loan Approval')
    axes[0, 1].set_xlabel('Loan Approved (0=No, 1=Yes)')
    axes[0, 1].set_ylabel('Credit Score')
    
    # Plot 3: Education level vs approval rate
    education_approval = df.groupby('education')['loan_approved'].mean().sort_values(ascending=False)
    education_approval.plot(kind='bar', ax=axes[1, 0], color='skyblue')
    axes[1, 0].set_title('Loan Approval Rate by Education Level')
    axes[1, 0].set_xlabel('Education Level')
    axes[1, 0].set_ylabel('Approval Rate')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Plot 4: Debt-to-income ratio distribution
    sns.histplot(data=df, x='debt_to_income_ratio', hue='loan_approved', 
                 multiple="layer", ax=axes[1, 1])
    axes[1, 1].set_title('Debt-to-Income Ratio Distribution by Loan Approval')
    axes[1, 1].set_xlabel('Debt-to-Income Ratio')
    
    plt.tight_layout()
    plt.savefig('loan_dataset_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nAnalysis plots saved as 'loan_dataset_analysis.png'")
    
except ImportError:
    print("\nMatplotlib/Seaborn not available. Skipping visualizations.")

# Generate a feature engineering helper
print("\n=== Feature Engineering Suggestions ===")
print("""
Suggested engineered features:
1. loan_to_income_ratio = loan_amount_requested / income
2. savings_to_loan_ratio = savings_balance / loan_amount_requested
3. credit_utilization = number_of_credit_cards * 5000 / income  # Assuming $5k limit per card
4. age_group = categorical bins (20-30, 30-40, 40-50, 50-60, 60+)
5. has_bankruptcy = 1 if bankruptcies > 0 else 0
6. has_late_payments = 1 if late_payments_6m > 0 else 0
7. employment_stability = employment_years / age
""")