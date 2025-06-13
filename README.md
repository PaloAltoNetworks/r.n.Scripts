# r.n.Scripts: Script Analysis Tool

## Overview

<p align="center">
  <img src="https://img.shields.io/badge/Powered%20by-Google%20Vertex%20AI-4285F4?style=for-the-badge&logo=google-cloud&logoColor=white" alt="Powered by Google Cloud Vertex AI"/>
  <img src="https://img.shields.io/badge/Language-Python%203.10%2B-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python Version"/>
  <img src="https://img.shields.io/badge/Licence-MIT-green?style=for-the-badge" alt="License: MIT"/>
</p>

`r.n.Scripts` is a command-line interface (CLI) tool designed to assist reverse engineers, security analysts, threat hunters, incident responders, and detection engineers in analyzing suspicious scripts and code. Leveraging Google's Vertex AI (specifically Gemini models) alongside static analysis techniques like Abstract Syntax Tree (AST) and Control Flow Graph (CFG) generation, `r.n.Scripts` automates the process of understanding script functionality, identifying malicious patterns, and even generating deobfuscation routines.

By providing various analysis perspectives ("prompts"), `r.n.Scripts` tailors its AI-driven insights to the specific needs of different security roles, significantly reducing manual effort and accelerating the speed of threat intelligence.

### Disclaimers: The tool should be run inside a safe Virtual Machine.

## Features

-   **AI-Powered Analysis**: Integrates with Google Vertex AI's Gemini models to provide contextual analysis of code snippets.
-   **Role-Specific Prompts**: Offers a range of pre-defined analysis prompts tailored for:
    -   General Malware Analysis
    -   SOC Analyst / Incident Responder
    -   Threat Hunter (with MITRE ATT&CK mapping focus)
    -   Detection Engineer (with Sigma rule generation focus)
    -   Configuration Extraction and Analysis
    -   C2 Communication / String Handling / Anti-Analysis
    -   Deobfuscation Code Generation (with Docker execution)
    -   High-level Code Summarization
-   **Static Code Analysis**:
    -   Generates Abstract Syntax Trees (AST) for structured code representation.
    -   Generates Control Flow Graphs (CFG) from ASTs for visualizing execution flow.
    -   Supports combined AST and CFG analysis for comprehensive insights.
-   **Noise Reduction**: Includes optional trimming and filtering mechanisms for AST and CFG to reduce verbosity and focus AI analysis on critical elements.
-   **Deobfuscation Code Generation & Execution**: For `Deobfuscation` prompts, `r.n.Scripts` attempts to generate Python `decoder.py` scripts and a `Dockerfile` for executing them in an isolated Docker container, providing the deobfuscated output.
-   **Interactive CLI**: Guides the user through analysis options using `questionary` for an intuitive experience.
-   **File Metadata & Utils**:
    -   Calculates SHA256 hash and Shannon Entropy of the input file.
    -   Identifies file types and mime-types using Magika.
-   **Token & Cost Estimation**: Provides real-time estimates of token usage and associated costs for LLM interactions.
-   **Comprehensive Output**: Generates detailed markdown reports (`.txt` files) containing LLM responses, Docker execution logs (if applicable), and raw AST/Graph data.
-   **Supported Languages**: Currently supports `python`, `javascript`, `powershell`, `php`, and `html` for AST/Graph generation and analysis. Other script types will fallback to code summary analysis if Tree-sitter grammar is not available.

## Requirements

### Software
-   **Python 3.10+**: The primary execution environment.
-   **pip**: Python package installer (usually comes with Python).
-   **Docker Desktop / Engine**: Required for the `Deobfuscation Code Generation` feature to execute generated Python deobfuscation scripts in a sandboxed environment.
-   **Google Cloud Project**: A Google Cloud Platform project is required to access Vertex AI services.

### Google Cloud Configuration
1.  **Enable Vertex AI API**: Ensure the Vertex AI API is enabled in your Google Cloud Project.
    -   Go to [Google Cloud Console](https://console.cloud.google.com/).
    -   Navigate to `APIs & Services` > `Enabled APIs & Services`.
    -   Search for "Vertex AI API" and enable it.
2.  **Authentication**: `r.n.Scripts` uses Application Default Credentials (ADC) to authenticate with Vertex AI.
    -   **Recommended for Local Development**: Use the `gcloud CLI`.
        ```bash
        gcloud auth application-default login
        ```
        This command will open a browser window for you to log in with your Google account.
    -   **For Production/CI/CD**: Use a Service Account key file. Set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to the path of your service account JSON key file.
        ```bash
        export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/keyfile.json"
        ```
3.  **Project ID and Region**: Update the `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION` variables in `.env` to match your Google Cloud Project ID and preferred Vertex AI region.

## Installation

1.  **Clone the Repository**:
    ```bash
    git clone https://gitlab.com/your-username/r.n.Scripts.git
    cd r.n.Scripts
    ```

2.  **Create a Virtual Environment (Recommended)**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies**:
    `r.n.Scripts` uses several external libraries. It's recommended to install them via `pip`.
    First, create a `requirements.txt` file (if not already present):

    ```text
    # requirements.txt
    vertexai>=1.41.0
    google-generativeai>=0.6.0
    networkx>=3.2.1
    tree-sitter>=0.21.0
    tree-sitter-languages>=1.4.0
    tree-sitter-python>=0.20.1
    tree-sitter-javascript>=0.20.2
    tree-sitter-powershell>=0.20.2
    tree-sitter-html>=0.20.2
    tree-sitter-php>=0.20.2
    docker>=7.0.0
    questionary>=2.0.0
    colorama>=0.4.6
    magika>=0.5.0
    scipy>=1.12.0
    pyfiglet>=1.0.0
    ```

    Then, install:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Run the Main Script**:
    ```bash
    python main.py
    ```

2.  **Follow the Prompts**:
    The CLI will guide you through the analysis process:

    -   **Enter the path to the file**: Provide the full path to the script or code file you want to analyze.
    -   **Choose a Gemini Model**: Select the Vertex AI Gemini model to use for analysis (e.g., `gemini-1.5-pro-002`). Note that different models have different pricing and capabilities.
    -   **Choose a System Instruction Persona**: Select the System Instructions (e.g. SOC Analyst, Threat Hunter, Incident Response, Detection Engineering, General Summary, Reverse Engineer )
    -   **Choose a Detailed Analysis Prompt:**: Select the analysis prompt (e.g. Focus: General Malware Analysis))
    -   **Choose Analysis Method**: Depending on the input file's language and capabilities, you might be asked to choose between `AST Only`, `Graph Only`, `AST and Graph Analysis`, or `Code Summary Only`. If Tree-sitter fails for the given file type, it will automatically fallback to `Code Summary Only`.
    -   **Apply AST/Graph trimming**: You'll be prompted to apply noise reduction for more focused analysis.
    -   **(Optional) Execute Python code in Docker container?**: If you selected a Deobfuscation prompt, and Docker is running, you'll be asked if you want to execute the generated deobfuscation script.
    -   **Choose a prompt for the Overall Code Summary Report:**: Select the summary prompt tailored to (e.g. SOC Analyst, Threat Hunter, Incident Response, Detection Engineering, General Summary, Reverse Engineer ))

### Example Workflow: Deobfuscation

1.  Run `python main.py`.
2.  Enter the path to your obfuscated script (e.g., `samples/obfuscated.ps1`).
3.  Choose a Gemini Model.
4.  Select `Focus: Deobfuscation Code Generation (Option: Execute Code in Docker Container)` as the prompt type.
5.  Select `Code Summary Only` or other relevant analysis method.
6.  If prompted, confirm the Docker execution.

`r.n.Scripts` will then interact with the LLM, receive the generated Python `decoder.py` and `Dockerfile`, build/run the Docker container, and output the deobfuscated content.

## Configuration

The `config.py`, `prompts.json`, `.env` file allows customization of parameters and analysis settings:

-   `vertexai.init`: **Crucially, update `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION`  to match your Google Cloud setup.
-   `INPUT_TOKEN_PRICING_PER_MILLION`: Defines pricing tiers for different Gemini models (for cost estimation).
-   `GENERATION_CONFIG`: Adjusts LLM generation parameters like `max_output_tokens`, `temperature`, and `top_p`.
-   `SAFETY_SETTING`: Controls content safety filters for LLM responses.
-   `USER_MODEL`: Defines the available Gemini models that can be selected.
-   `SYSTEM_INSTRUCTION_SUMMARY_PERSONAS`: Contains system instruction personas for different operational roles.
-   `PROMPT_TEMPLATES`: Contains the detail prompt instruction for different operational roles or areas of focus.
-   `SUMMARY_PROMPT_TEMPLATES`: Contains the summary prompts tailored to specific operational roles.


## Output

`r.n.Scripts` generates several files in the directory where it's executed:

-   `<SHA256_HASH>_<MODEL_NAME>_Report.txt`: The primary report file, containing:
    -   Initial file metadata (SHA256, Entropy, Magika type).
    -   Selected model, prompt, and analysis method.
    -   The comprehensive LLM-generated analysis response.
    -   If Docker execution was performed, the raw output from the Docker container.
    -   A final, high-level code summary.
-   `<SHA256_HASH>_AST_out.txt`: If AST analysis was performed, this file contains the JSON representation of the Abstract Syntax Tree.
-   `<SHA256_HASH>_GRAPH_out.txt`: If Graph analysis was performed, this file contains the JSON representation of the Control Flow Graph (NetworkX node-link format).
-   `decoder.py`: If the `Deobfuscation Code Generation` prompt is used AND the LLM successfully generates Python code, this file will contain the generated deobfuscation script.
-   `<SHA256_HASH>_rnScript.log`: A detailed log file of the application's execution, useful for debugging.

## Contributing

Contributions are welcome! If you have suggestions for new features, bug fixes, or improved prompt templates, please open an issue or submit a merge request.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.