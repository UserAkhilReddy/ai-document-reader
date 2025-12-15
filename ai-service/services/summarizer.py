from transformers import pipeline
summarizer_pipeline = pipeline("summarization", model = "facebook/bart-large-cnn")

def summarize_chunks(chunks):
    """ takes all chunks and returns final summary"""

    summaries = []
    
    for chunk in chunks:
        if len(chunk.split())<50:
            continue
        result = summarizer_pipeline(chunk, max_length= 130, min_length= 40, do_sample= False )

        summaries.append(result[0]["summary_text"])

        final_summary = " ".join(summaries)

        return   final_summary
