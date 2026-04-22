from parser import parse_resume # Your friend's code
from ai_evaluator import generate_evaluation_report # Your code

def run_test():
    print("--- STARTING INTEGRATION TEST ---")
    
    # 1. Provide a dummy job description
    sample_jd = """
    Looking for a Python Backend Developer. 
    Required skills: Python, REST APIs, JSON data handling, basic NLP or AI integration experience.
    Must be able to write clean, modular code.
    """
    
    # 2. Run the parser (Make sure you have a sample resume.pdf in your folder)
    print("Step 1: Parsing PDF...")
    try:
        parsed_data = parse_resume("resume.pdf")
        print(f"Successfully extracted data for: {parsed_data.get('name')}")
    except Exception as e:
        print(f"Failed to parse resume: {e}")
        return

    # 3. Run your evaluator
    print("\nStep 2: Sending to Gemini and Generating Report...")
    output_file = generate_evaluation_report(
        parsed_resume_data=parsed_data, 
        job_description=sample_jd,
        output_filename="Test_Result.docx"
    )
    
    if output_file:
        print(f"\nSUCCESS! Open {output_file} to see the result.")
    else:
        print("\nFAILED to generate report.")

if __name__ == "__main__":
    run_test()