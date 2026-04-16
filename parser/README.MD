Parser Module

This module is responsible for reading resumes and extracting useful information from them.

It supports both PDF and DOCX files and converts the resume content into structured data that can be used by other parts of the project.

Files

- parser.py – main logic for parsing resumes
- test.py – used to test the parser
- requirements.txt – required libraries

Working

1. The resume file is read and converted into plain text.
2. Email and phone number are extracted using simple patterns.
3. Skills are identified from the text using basic NLP techniques.
4. Education and experience sections are detected in a basic way.
5. A predicted role is generated based on the extracted skills.

Main Function

parse_resume(file_path)

This function takes a resume file as input and returns the extracted data in dictionary format.

Sample Output

{
  "name": "John Doe",
  "email": "john@gmail.com",
  "phone": "9876543210",
  "skills": ["python", "sql"],
  "education": {},
  "experience": [],
  "predicted_role": "Data Scientist"
}

Summary

The parser converts unstructured resume data into a structured format which is used by the rest of the system.
