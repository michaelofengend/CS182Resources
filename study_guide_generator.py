import os
import re
import datetime
import google.generativeai as genai
import time
import argparse
from pathlib import Path

# Configuration
BASE_DIR = Path('/Users/michaelofengenden/Desktop/CS182Notes')
LECTURE_NOTES_DIR = BASE_DIR / 'Lecture Notes'
SRTS_DIR = BASE_DIR / 'SRTsFromLecture'
DISCUSSIONS_DIR = BASE_DIR / 'Discussions'
OUTPUT_DIR = BASE_DIR / 'StudyGuides'

# Ensure output directory exists
OUTPUT_DIR.mkdir(exist_ok=True)

# Regex patterns
DATE_PATTERN = re.compile(r'(\d{1,2})[⧸/](\d{1,2})[⧸/](\d{4})') 
LECTURE_NUM_PATTERN = re.compile(r'Lecture\s*(\d+)')
DISC_NUM_PATTERN = re.compile(r'dis(\d+)')

def get_srt_files():
    files = []
    # Handle the weird slash character by checking glob logic or generic iteration
    # The ls showed specific unicode chars, so we iterate all .srt
    for f in SRTS_DIR.glob('*.srt'):
        match = DATE_PATTERN.search(f.name)
        if match:
            month, day, year = map(int, match.groups())
            dt = datetime.date(year, month, day)
            files.append({'path': f, 'date': dt})
        else:
            print(f"Warning: Could not parse date from SRT: {f.name}")
    return sorted(files, key=lambda x: x['date'])

def get_lecture_files():
    files = []
    for f in LECTURE_NOTES_DIR.glob('*.pdf'):
        match = LECTURE_NUM_PATTERN.search(f.name)
        if match:
            num = int(match.group(1))
            files.append({'path': f, 'num': num})
        else:
            # Handle "Lecture1Ranade.pdf" -> Lecture 1
            if "Lecture1" in f.name:
                 files.append({'path': f, 'num': 1})
            else:
                print(f"Warning: Could not parse number from Lecture Note: {f.name}")
    return sorted(files, key=lambda x: x['num'])

def get_discussion_files():
    files = []
    for f in DISCUSSIONS_DIR.glob('*.pdf'):
        match = DISC_NUM_PATTERN.search(f.name)
        if match:
            num = int(match.group(1))
            files.append({'path': f, 'num': num})
        else:
            print(f"Warning: Could not parse number from Discussion: {f.name}")
    return sorted(files, key=lambda x: x['num'])

def create_groups(srts, notes, discussions):
    groups = []
    max_len = max(len(srts), len(notes))
    
    for i in range(max_len):
        group = {
            'id': i + 1,
            'srt': srts[i]['path'] if i < len(srts) else None,
            'note': notes[i]['path'] if i < len(notes) else None,
            'discussions': []
        }
        
        if group['note']:
            lecture_num = notes[i]['num']
            # Map discussion: Lecture N -> Discussion ceil(N/2)
            # Lecture 1,2 -> Disc 1 (Idx 0)
            # Lecture 3,4 -> Disc 2 (Idx 1)
            disc_idx = (lecture_num + 1) // 2 - 1
            if 0 <= disc_idx < len(discussions):
                group['discussions'].append(discussions[disc_idx]['path'])
                
        groups.append(group)
    return groups

def upload_and_wait(path, mime_type=None):
    if not path:
        return None
    print(f"Uploading {path.name}...")
    file_ref = genai.upload_file(path, mime_type=mime_type)
    
    # Wait for processing
    while file_ref.state.name == "PROCESSING":
        print('.', end='', flush=True)
        time.sleep(2)
        file_ref = genai.get_file(file_ref.name)
        
    if file_ref.state.name == "FAILED":
        print(f"File upload failed: {path.name}")
        return None
    
    print(f"Ready: {path.name}")
    return file_ref

def generate_study_guide(group, model):
    print(f"\nProcessing Group {group['id']}...")
    
    uploaded_files = []
    
    # Upload Note
    note_file = upload_and_wait(group['note'], mime_type='application/pdf')
    if note_file: uploaded_files.append(note_file)
    
    # Upload SRT
    # SRT is text, but can be uploaded as text/plain or processed as context.
    # Uploading as file is cleaner for large context.
    srt_file = upload_and_wait(group['srt'], mime_type='text/plain')
    if srt_file: uploaded_files.append(srt_file)
    
    # Upload Discussions
    for disc_path in group['discussions']:
        disc_file = upload_and_wait(disc_path, mime_type='application/pdf')
        if disc_file: uploaded_files.append(disc_file)
        
    if not uploaded_files:
        print("No files to process for this group.")
        return

    prompt = """
    You are an expert Professor and Tutor for CS 182 (Neural Networks).
    I have provided you with the Lecture Notes, the Lecture Transcript (SRT), and the relevant Discussion Worksheet for a specific topic.

    Your goal is to create a **Thorough Study Guide** for this material.
    
    **CRITICAL INSTRUCTIONS:**
    1.  **Intuition First**: For every concept introduced, first explain the *intuition* behind it. Why do we need it? What problem does it solve?
    2.  **Analogies**: Provide easy-to-follow, real-world analogies for technical terms and complex math. (e.g., explaining Gradient Descent as walking down a hill, but more creative and specific to the nuance).
    3.  **Synthesis**: Do not just summarize each file separately. Synthesize the Lecture Notes (theory) with the Transcript (prof's explanation) and the Discussion (practice).
    4.  **Structure**:
        *   **Core Concepts**: deep dive into the main ideas.
        *   **Key Analogies**: explicit section for analogies.
        *   **Math Decoded**: explain the equations in plain English.
        *   **Practice Insights**: Use the discussion worksheet to show how these concepts are applied to problems.
    
    Make the tone encouraging but rigorous. Assume the student is smart but new to the material.
    """
    
    try:
        response = model.generate_content(
            [prompt, *uploaded_files],
            request_options={"timeout": 600} # 10 minute timeout
        )
        
        output_filename = f"Study_Guide_Lecture_{group['id']:02d}.md"
        output_path = OUTPUT_DIR / output_filename
        
        with open(output_path, 'w') as f:
            f.write(response.text)
            
        print(f"Generated: {output_path}")
        
    except Exception as e:
        print(f"Error generating content: {e}")
    finally:
        # Cleanup files to avoid hitting limits (optional, but good practice)
        for f in uploaded_files:
            try:
                genai.delete_file(f.name)
            except:
                pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, help='Limit number of groups to process')
    args = parser.parse_args()

    if not os.environ.get("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY not found in environment.")
        return

    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    
    # Use gemini-2.0-flash-exp for best performance/speed/multimodal
    model = genai.GenerativeModel('models/gemini-2.0-flash-exp')
    
    print("Scanning files...")
    srts = get_srt_files()
    notes = get_lecture_files()
    discs = get_discussion_files()
    
    groups = create_groups(srts, notes, discs)
    
    # Filter groups if limit is set
    if args.limit:
        groups = groups[:args.limit]
    
    for group in groups:
        output_filename = f"Study_Guide_Lecture_{group['id']:02d}.md"
        output_path = OUTPUT_DIR / output_filename
        
        if output_path.exists():
            print(f"Skipping Group {group['id']} (File exists: {output_filename})")
            continue
            
        generate_study_guide(group, model)
        time.sleep(5) # Rate limiting pause

if __name__ == "__main__":
    main()
