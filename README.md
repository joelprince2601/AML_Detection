# AML Detection

This project is designed to detect Acute Myeloid Leukemia (AML) using machine learning techniques. It processes input data, applies predictive models, and outputs detection results.

# AML Hosted in Streamlit
use the below link to access the application and try it out yourself with the sample dataset which is provided in the website.
https://amldetectionjp.streamlit.app/

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/joelprince2601/AML_Detection.git
   cd AML_Detection
   ```

2. **Install the required dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Prepare your data:**

   Ensure your input data is in the correct format as expected by the system. You can use the `create_example_data.py` script to generate example data.

2. **Process the data:**

   Use the `excel_processor.py` script to process your input data. This script will prepare the data for the detection model.

3. **Run the AML detection:**

   Execute the `run_aml_detection.bat` script to run the detection model on your processed data.

## Project Structure

```plaintext
AML_Detection/
├── aml_detection/             # Core detection module
├── aml_detection.egg-info/    # Package metadata
├── temp/                      # Temporary files
├── create_example_data.py     # Script to create example data
├── excel_processor.py         # Script to process Excel data
├── project_structure.txt      # Text file outlining the project structure
├── requirements.txt           # List of required Python packages
├── run_aml_detection.bat      # Batch script to run AML detection
└── setup.py                   # Setup script for packaging
```

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes. Ensure that your code adheres to the project's coding standards and includes appropriate tests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

