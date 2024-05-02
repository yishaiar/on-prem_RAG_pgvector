

from app import BASE_DIR  # simply because we need to load .env
from app.db.connect import create_db_connection
from tqdm.auto import tqdm
import os
import psycopg2
# from openai.embeddings_utils import get_embeddings
from embedding_models.hugginface import load_LLM,embed

def VectorIndexUpdate(texts,tokenizer= None,model= None,device = None):
    LEN,batch_size = len(texts),int(os.getenv('batch_size'))
        

    batch_size =1
    for i in tqdm(range(0, LEN, batch_size)):

        # start = time.time()   
        texts_batch = texts[i:min(i+batch_size,LEN)].copy()


        
        # embeddings_batch = get_embeddings(texts_batch, os.getenv('OPENAI_EMBEDDING_MODEL'),
        #                          api_key = os.getenv('OPENAI_API_KEY'))
        embeddings_batch = embed(texts_batch,tokenizer=tokenizer,model=model,device=device)
    
    
    

        # Write text and embeddings to database using Psycopg; a python adapter for PostgreSQL database 
        connection = create_db_connection()
        cursor = connection.cursor()
        try:
            for text, embedding in zip(texts_batch, embeddings_batch):
                cursor.execute(
                    "INSERT INTO embeddings (embedding, content) VALUES (%s, %s)",
                    (embedding, text.replace("\x00", "\uFFFD"))
                    # (list(embedding.astype(float)), text)
                )
            connection.commit()
        except (Exception, psycopg2.Error) as error:
            print("Error while writing to DB", error)
            if 'NUL' in str(error):
                # print(text)
                return text, embedding
        finally:
            if cursor:
                cursor.close()
            if connection:
                connection.close()
            # print(i)
        # if i>10:
        #     break
    print("finished writing Data to DB!")
if __name__ == '__main__':

    # Write five example sentences that will be converted to embeddings
    texts = [
        "I like to eat broccoli and bananas.",
        "I ate a banana and spinach smoothie for breakfast.",
        "Chinchillas and kittens are cute.",
        "My sister adopted a kitten yesterday.",
        "Look at this cute hamster munching on a piece of broccoli.",
    ]
    model, tokenizer, device = load_LLM(os.getenv('model_id'))
    VectorIndexUpdate(texts,tokenizer=tokenizer,model=model,device=device)

