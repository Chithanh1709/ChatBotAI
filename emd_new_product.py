# file: add_to_chroma.py

import json
import hashlib
from sentence_transformers import SentenceTransformer
import chromadb

class ChromaDataInserter:
    def __init__(self, db_path, collection_name):
        self.db_path = db_path
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        
        # Load model
        print("Loading Vietnamese embedding model...")
        self.model = SentenceTransformer(
            "keepitreal/vietnamese-sbert",
            device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
        )
    
    def generate_chunk_id(self, text, metadata):
        """Generate unique ID based on content"""
        content = f"{text}_{metadata.get('product_id', '')}"
        return f"chunk_{hashlib.md5(content.encode()).hexdigest()[:12]}"
    
    def connect_to_database(self):
        """Connect to existing ChromaDB"""
        print("Connecting to ChromaDB...")
        try:
            self.client = chromadb.PersistentClient(path=self.db_path)
            self.collection = self.client.get_collection(name=self.collection_name)
            print(f"Connected to collection: {self.collection_name}")
            return True
        except Exception as e:
            print(f"Connection error: {e}")
            return False
    
    def prepare_batch_data(self, data_list):
        """Prepare batch data for database insertion"""
        print("Creating embeddings...")
        
        texts = [item["text"] for item in data_list]
        metadatas = [item["metadata"] for item in data_list]
        
        # Create batch embeddings
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).tolist()
        
        # Generate IDs
        ids = [self.generate_chunk_id(text, metadata) for text, metadata in zip(texts, metadatas)]
        
        return {
            "ids": ids,
            "embeddings": embeddings,
            "documents": texts,
            "metadatas": metadatas
        }
    
    def check_existing_ids(self, ids):
        """Check which IDs already exist"""
        print("Checking for duplicate data...")
        existing_ids = set()
        
        try:
            existing_data = self.collection.get(include=[])
            existing_ids = set(existing_data['ids'])
            print(f"Found {len(existing_ids)} existing documents in database")
        except Exception as e:
            print(f"Cannot get ID list: {e}")
        
        return existing_ids
    
    def add_batch_products(self, data_list):
        """Add multiple products at once"""
        if not self.connect_to_database():
            return False
        
        try:
            # Prepare batch data
            batch_data = self.prepare_batch_data(data_list)
            
            # Check for duplicates
            existing_ids = self.check_existing_ids(batch_data["ids"])
            
            # Filter only new data
            new_indices = [i for i, id_ in enumerate(batch_data["ids"]) if id_ not in existing_ids]
            
            if not new_indices:
                print("All documents already exist, nothing to add")
                return {"added": 0, "existed": len(data_list)}
            
            # Filter new data
            new_ids = [batch_data["ids"][i] for i in new_indices]
            new_embeddings = [batch_data["embeddings"][i] for i in new_indices]
            new_documents = [batch_data["documents"][i] for i in new_indices]
            new_metadatas = [batch_data["metadatas"][i] for i in new_indices]
            
            print(f"Will add {len(new_ids)}/{len(data_list)} new documents")
            
            # Add to database in batches
            chunk_size = 500
            total_added = 0
            
            for i in range(0, len(new_ids), chunk_size):
                end_idx = min(i + chunk_size, len(new_ids))
                
                self.collection.add(
                    ids=new_ids[i:end_idx],
                    embeddings=new_embeddings[i:end_idx],
                    documents=new_documents[i:end_idx],
                    metadatas=new_metadatas[i:end_idx]
                )
                
                batch_count = end_idx - i
                total_added += batch_count
                print(f"Added batch {i//chunk_size + 1} ({batch_count} documents)")
            
            # Statistics
            stats = {
                "added": total_added,
                "existed": len(data_list) - total_added,
                "total_processed": len(data_list)
            }
            
            self.print_batch_statistics(stats, data_list)
            return stats
            
        except Exception as e:
            print(f"Error adding batch: {e}")
            return {"added": 0, "existed": 0, "error": str(e)}
    
    def add_from_json_file(self, json_file_path):
        """Add data from JSON file"""
        print(f"Reading data from {json_file_path}...")
        
        try:
            with open(json_file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Handle both list and single object cases
            if isinstance(data, dict):
                data_list = [data]
            elif isinstance(data, list):
                data_list = data
            else:
                print("Invalid JSON format")
                return
            
            print(f"Read {len(data_list)} items from file")
            return self.add_batch_products(data_list)
            
        except Exception as e:
            print(f"Error reading file: {e}")
            return {"added": 0, "existed": 0, "error": str(e)}
    
    def print_batch_statistics(self, stats, data_list):
        """Print statistics after adding data"""
        print("\nADD DATA STATISTICS:")
        print(f"   Total items processed: {stats['total_processed']}")
        print(f"   Newly added: {stats['added']}")
        print(f"   Already existed: {stats['existed']}")
        
        # Product statistics
        product_ids = set()
        categories = set()
        for item in data_list:
            metadata = item.get("metadata", {})
            product_ids.add(metadata.get('product_id', ''))
            categories.add(metadata.get('category', ''))
        
        print(f"   Unique products: {len([pid for pid in product_ids if pid])}")
        print(f"   Categories: {len([c for c in categories if c])}")
        
        # Total documents in database
        try:
            total_in_db = self.collection.count()
            print(f"   Total documents in database: {total_in_db}")
        except:
            pass

def add_json_to_chroma(json_file_path, db_path, collection_name):
    """Quick function to add data from JSON file"""
    inserter = ChromaDataInserter(db_path, collection_name)
    return inserter.add_from_json_file(json_file_path)

# Main execution
if __name__ == "__main__":
    # Initialize inserter
    inserter = ChromaDataInserter(
        db_path="D:/chroma_food_rag",
        collection_name="food_products_vn"
    )
    
    # Add data from JSON file
    result = inserter.add_from_json_file("rag_chunks_moi.json")
    print(f"Result: {result}")