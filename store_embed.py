import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
import os

# Define dataset folder and initialize chromadb client
dataset_folder = 'Data'
chroma_client = chromadb.PersistentClient(path="Vector_database")
image_loader = ImageLoader()
CLIP = OpenCLIPEmbeddingFunction()

# Create or get the collection
image_vdb = chroma_client.get_or_create_collection(name="image", embedding_function=CLIP, data_loader=image_loader)

# Initialize lists to store ids and uris
ids = []
uris = []

# Loop through images and append ids and uris
for i, filename in enumerate(sorted(os.listdir(dataset_folder))):
    if filename.endswith('.png'):
        file_path = os.path.join(dataset_folder, filename)
        
        ids.append(str(i))  # Store id as string
        uris.append(file_path)  # Store file path (URI)

# Set the batch size limit (e.g., 166 to avoid exceeding the limit)
batch_size = 166  # Maximum allowed batch size (as per your error message)

# Add images to the vector database in smaller batches
for i in range(0, len(ids), batch_size):
    batch_ids = ids[i:i + batch_size]
    batch_uris = uris[i:i + batch_size]
    
    # Add the current batch to the vector database
    image_vdb.add(
        ids=batch_ids,
        uris=batch_uris
    )

print("Images stored to the Vector database.")
