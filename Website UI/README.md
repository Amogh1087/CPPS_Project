This folder contains the UI of the website. It consists of three main parts that work in the following order:
1. Homepage.html - The landing page users see when they open the website. It includes options such as resume analysis, feature overview, workflow explanation, and access to saved reports.
2. Analyzer.html - The core functionality of the website. Users upload their resume and select a target job role (currently supports only .pdf and .docx formats). The system analyzes the resume and redirects to the report
   page.
3. Report.html - Displays a structured analysis of how well the resume aligns with the selected job role, along with actionable suggestions for improvement and missing elements that could improve chances of being hired.

Reports are updated each time the same resume is re-uploaded, maintaining a 1:1 mapping between a resume and its report. Each new resume generates a separate report, which includes a timestamp indicating the latest update. Currently, reports are not being displayed due to a known issue. This is under active development and will be resolved soon.
