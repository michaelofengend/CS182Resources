# CS 182 Study Guide Generator

This tool automates the creation of intuition-focused study guides for CS 182 (Neural Networks) by synthesizing three data sources:
1.  **Lecture Notes** (PDFs)
2.  **Lecture Transcripts** (SRT files)
3.  **Discussion Solution Worksheets** (PDFs)

It uses the Gemini API (multimodal capabilities) to process these files directly and generate a comprehensive markdown study guide for each lecture topic.

## Features

-   **Intelligent Mapping**: Automatically aligns lecture notes (by number) with transcripts (by date) and discussion worksheets.
-   **Multimodal Processing**: Uploads PDFs and text directly to Gemini for high-fidelity context window utilization.
-   **Intuition-First Output**: Prompts the model to focus on intuitive explanations and real-world analogies before diving into the math.
-   **Resume Capability**: Skips lectures that already have a generated study guide in the output folder.

## Setup

1.  **Install Dependencies**:
    You need the Google Generative AI Python SDK.
    ```bash
    pip install google-generativeai
    ```

2.  **API Key**:
    Set your Gemini API key as an environment variable.
    ```bash
    export GEMINI_API_KEY='your_api_key_here'
    ```

## Usage

Run the script from the directory containing your notes:

```bash
python study_guide_generator.py
```

### Options

-   `--limit N`: Limit the run to the first N groups (useful for testing).
    ```bash
    python study_guide_generator.py --limit 1
    ```

## Output

Generated study guides are saved in the `StudyGuides/` directory as markdown files (e.g., `Study_Guide_Lecture_01.md`).
