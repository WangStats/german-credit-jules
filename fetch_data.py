import requests
import zipfile
import io
import pandas as pd
import os

def fetch_and_process_data():
    url = "https://archive.ics.uci.edu/static/public/144/statlog+german+credit+data.zip"
    print(f"Downloading data from {url}...")
    response = requests.get(url)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        print("Extracting german.data...")
        with z.open('german.data') as f:
            lines = f.readlines()

    data = []
    for line in lines:
        parts = line.decode('utf-8').strip().split()
        if not parts:
            continue
        # Check if we have 20 attributes + 1 class
        if len(parts) != 21:
            print(f"Skipping line with unexpected number of columns: {len(parts)}")
            continue

        # Attributes indices (0-based) from 1-based documentation
        # A1: 0, A2: 1, ... A20: 19, Class: 20

        # Mapping
        # Checking account (Attribute 1)
        checking_status = parts[0]
        checking_map = {
            'A11': '< 0 DM',
            'A12': '0 <= ... < 200 DM',
            'A13': '>= 200 DM / salary assignments for at least 1 year',
            'A14': 'no checking account'
        }
        checking = checking_map.get(checking_status, checking_status)

        # Duration (Attribute 2)
        duration = int(parts[1])

        # Purpose (Attribute 4)
        purpose_status = parts[3]
        purpose_map = {
            'A40': 'car (new)',
            'A41': 'car (used)',
            'A42': 'furniture/equipment',
            'A43': 'radio/television',
            'A44': 'domestic appliances',
            'A45': 'repairs',
            'A46': 'education',
            'A47': 'vacation',
            'A48': 'retraining',
            'A49': 'business',
            'A410': 'others'
        }
        purpose = purpose_map.get(purpose_status, purpose_status)

        # Credit amount (Attribute 5)
        credit_amount = int(parts[4])

        # Saving accounts (Attribute 6)
        savings_status = parts[5]
        savings_map = {
            'A61': '< 100 DM',
            'A62': '100 <= ... < 500 DM',
            'A63': '500 <= ... < 1000 DM',
            'A64': '>= 1000 DM',
            'A65': 'unknown/ no savings account'
        }
        savings = savings_map.get(savings_status, savings_status)

        # Sex (Attribute 9)
        # A91 : male : divorced/separated
        # A92 : female : divorced/separated/married
        # A93 : male : single
        # A94 : male : married/widowed
        # A95 : female : single
        sex_status = parts[8]
        if sex_status in ['A91', 'A93', 'A94']:
            sex = 'male'
        elif sex_status in ['A92', 'A95']:
            sex = 'female'
        else:
            sex = 'unknown'

        # Age (Attribute 13)
        age = int(parts[12])

        # Housing (Attribute 15)
        housing_status = parts[14]
        housing_map = {
            'A151': 'rent',
            'A152': 'own',
            'A153': 'free'
        }
        housing = housing_map.get(housing_status, housing_status)

        # Job (Attribute 17)
        job_status = parts[16]
        job_map = {
            'A171': 'unskilled and non-resident',
            'A172': 'unskilled and resident',
            'A173': 'skilled',
            'A174': 'highly qualified'
        }
        job = job_map.get(job_status, job_status)

        # Risk (Attribute 21 / Class)
        risk_status = parts[20]
        if risk_status == '1':
            risk = 'good'
        elif risk_status == '2':
            risk = 'bad'
        else:
            risk = risk_status

        record = {
            'Age': age,
            'Sex': sex,
            'Job': job,
            'Housing': housing,
            'Saving accounts': savings,
            'Checking account': checking,
            'Credit amount': credit_amount,
            'Duration': duration,
            'Purpose': purpose,
            'Risk': risk
        }
        data.append(record)

    df = pd.DataFrame(data)

    # Fill missing values if any (though dataset shouldn't have any per documentation)
    # If any column has missing values, we can fill them with mode or mean, or a placeholder.
    # Since the columns are specific types, let's check.
    if df.isnull().values.any():
        print("Found missing values. Handling them...")
        for col in df.columns:
            if df[col].isnull().any():
                if df[col].dtype == 'object':
                    df[col] = df[col].fillna(df[col].mode()[0])
                else:
                    df[col] = df[col].fillna(df[col].median())

    output_file = 'german_credit_data.csv'
    print(f"Saving to {output_file}...")
    # Reorder columns to match the target file exactly
    columns = ['Age', 'Sex', 'Job', 'Housing', 'Saving accounts', 'Checking account', 'Credit amount', 'Duration', 'Purpose', 'Risk']
    df = df[columns]
    df.to_csv(output_file, index=False)
    print("Done.")

if __name__ == "__main__":
    fetch_and_process_data()
