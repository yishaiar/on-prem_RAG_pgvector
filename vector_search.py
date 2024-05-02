from app.db.connect import create_db_connection

import os
# from openai.embeddings_utils import get_embeddings
from embedding_models.hugginface import load_LLM,embed

def query(texts, model, tokenizer, device):
    # embedding = get_embedding(text, os.getenv('OPENAI_EMBEDDING_MODEL'))
    embeddings = embed(texts,tokenizer=tokenizer,model=model,device=device)

    connection = create_db_connection()
    cursor = connection.cursor()
    try:
        # cursor.execute(f"""SELECT * FROM embeddings""")
        for text, embedding in zip(texts, embeddings):
            cursor.execute(f"""
                SELECT content,  1 - (embedding <=> '{embedding}') AS cosine_similarity, id
                FROM embeddings
                ORDER BY cosine_similarity desc
                LIMIT {os.getenv('result_limit')}
            """)
            print (f'Input text: {text}')
            for r in cursor.fetchall():
                print(f"Similarity: {r[1]};\ttable ID: {r[2]}\tContent: {r[0]}")
            print('')
    except Exception as error:
        print("Error: ", error)
    finally:
        cursor.close()
        connection.close()
        
    print('finished reading Data from DB!')

if __name__ == '__main__':
    # This script is used to test the embedding model, and the
    # cosine similarity function within the database.
    
    model, tokenizer, device = load_LLM(os.getenv('model_id'))
    texts = [
        "Did anyone adopt a cat this weekend?",
        "What is the cutest animal?",
        ]
    query(texts, model, tokenizer, device)
    
    

