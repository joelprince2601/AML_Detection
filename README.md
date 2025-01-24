# AML Transaction Analysis System

The **AML Transaction Analysis System** is designed for transaction data analysis, specifically for detecting suspicious patterns indicative of money laundering activities. Upload your transaction data or use the provided example dataset for analysis.

# AML Hosted in Streamlit
Use the below link to access the application and try it out yourself with the sample dataset which is provided in the website.
https://amldetectionjp.streamlit.app/

## Features

- Analyze transaction data in Excel format.
- Identify unusual patterns or anomalies in transaction activities.
- User-friendly interface for data upload and processing.

## Table of Contents

- [Usage](#usage)
- [Data Requirements](#data-requirements)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Usage

1. **Upload Transaction Data:**
   - Upload your Excel file containing transaction data.
   - Alternatively, use the example dataset provided with the project.

2. **Run the Analysis:**
   - The system processes the uploaded data and provides insights into transaction anomalies.

3. **Review the Results:**
   - Examine the flagged transactions for further investigation.

## Data Requirements

The uploaded Excel file must include the following columns:

- `transaction_date` – Date of the transaction.
- `amount` – Transaction amount.
- `sender_id` – Unique ID of the sender.
- `sender_country` – Country of the sender.
- `recipient_id` – Unique ID of the recipient.
- `recipient_country` – Country of the recipient.

Example data format:

| transaction_date | amount | sender_id | sender_country | recipient_id | recipient_country |
|------------------|--------|-----------|----------------|--------------|-------------------|
| 2024-01-01       | 1000   | 12345     | USA            | 67890        | UK                |

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/joelprince2601/AML_Detection.git
   cd AML_Detection
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**

   ```bash
   python run_analysis.py
   ```

## Project Structure

```plaintext
AML_Transaction_Analysis/
├── example_dataset.xlsx       # Example transaction dataset
├── transaction_analysis/      # Core analysis module
├── requirements.txt           # Python dependencies
├── upload_processor.py        # Handles data uploads
├── run_analysis.py            # Main script to run analysis
└── README.md                  # Project documentation
```

## Contributing

Contributions are welcome! Fork the repository and submit a pull request with enhancements or fixes. Ensure your code adheres to the project standards.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
