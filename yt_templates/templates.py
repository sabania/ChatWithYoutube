INITIAL_TEMPLATE = """Answer the question below using the transcript as source. If the transcript does not contain any specific information, answer with "No information available in the source documents."
Do not give just short answers, but try to incorporate the information from all trancrtipts to give a complete answer and additonal information if possible.
Do not forget to add the time stamp as reference to the answer.
-----------
Transcripts:

{source_documents}

-----------
Answer the question using the time stamp as reference and the following format:
answer_part [time stamp] anser_part .... answer_part [time stamp] etc...
example answer: "The company is based in San Francisco. [2496.52] The company was founded in 2015. [3025.12] There are 100 employees."

------------
Questions:

{question}"""

FOLLOW_UP_TEMPLATE = """Transcripts:

{source_documents}

-----------

Question:

{question}"""


DOCUMENT_TEMPLATE = """Transcript {document_number}:
{document_content}"""

def get_initial_template(source_documents, question):
    doc_msg = "\n\n".join([DOCUMENT_TEMPLATE.format(document_number=i+1, document_content=doc.page_content) for i, doc in enumerate(source_documents)])
    return INITIAL_TEMPLATE.format(source_documents=doc_msg, question=question)

def get_follow_up_template(source_documents, question):
    doc_msg = "\n\n".join([DOCUMENT_TEMPLATE.format(document_number=i+1, document_content=doc.page_content) for i, doc in enumerate(source_documents)])
    return FOLLOW_UP_TEMPLATE.format(source_documents=doc_msg, question=question)



