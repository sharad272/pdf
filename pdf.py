from transformers import pipeline
import fitz  # PyMuPDF

# Load the Question-Answering Pipeline
model_name = "deepset/roberta-base-squad2"
qa_pipeline = pipeline('question-answering', model=model_name, tokenizer=model_name)

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file and split it into pages.
    """
    doc = fitz.open(pdf_path)
    text_chunks = []
    for page in doc:
        text = page.get_text()
        if text.strip():  # Only add non-empty pages
            text_chunks.append(text.strip())
    return text_chunks

def split_into_paragraphs(text):
    """
    Split text into paragraphs using logical delimiters.
    """
    return text.split('\n\n')  # Split by double newlines (common paragraph delimiter)

def generate_answer(question, document_pages, qa_pipeline, confidence_threshold=0.1):
    """
    Answer a question by iterating over document paragraphs and generating the top response.
    """
    top_result = None
    top_page = None
    top_paragraph = None
    top_score = float('-inf')

    for page_num, page_text in enumerate(document_pages):
        paragraphs = split_into_paragraphs(page_text)

        for paragraph_num, paragraph_text in enumerate(paragraphs):
            print(f"\nProcessing Page {page_num + 1}:")
          #  print(paragraph_text)  # Debug: Print the paragraph being processed

            # Use the pipeline for question answering
            try:
                result = qa_pipeline(question=question, context=paragraph_text)
            #    print(f"Raw Result: {result}")  # Debug: Inspect raw model output
            except Exception as e:
                print(f"Error processing paragraph {paragraph_num + 1} on page {page_num + 1}: {e}")
                continue

            # Extract answer and confidence score
            answer = result['answer']
            score = result['score']

            if score < confidence_threshold:
            #    print(f"Low confidence ({score}). Skipping paragraph.")  # Debug: Low confidence
                continue

            # Update the top result
            if score > top_score:
                top_result = answer
                top_page = page_num + 1
                top_paragraph = paragraph_num + 1
                top_score = score

    return f"Page {top_page}, Paragraph {top_paragraph}: {top_result} (Confidence: {top_score:.2f})" if top_result else "No valid answer found."

# Main Execution
if __name__ == "__main__":
    pdf_path = "medical.pdf"  # Replace with your PDF file path
    texts = extract_text_from_pdf(pdf_path)
    print(f"Extracted {len(texts)} pages from the PDF.\n")

    while True:
        user_query = input("Enter your query (or type 'exit' to quit): ").strip()
        if user_query.lower() == 'exit':
            print("Exiting...")
            break

        # Treat user query as-is
        answer = generate_answer(user_query, texts, qa_pipeline)
        print("\nAnswer:")
        print(answer)
