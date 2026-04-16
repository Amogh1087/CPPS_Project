from parser import parse_resume

print("PDF OUTPUT:")
print(parse_resume("resume.pdf"))

print("\nDOCX OUTPUT:")
print(parse_resume("resume.docx"))
