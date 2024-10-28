
import os
from google.cloud import storage
from google.cloud import aiplatform

# Set up Google Cloud credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'path/to/your/credentials.json'

def upload_model_to_gcs(bucket_name, source_file_name, destination_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print(f"File {source_file_name} uploaded to {destination_blob_name}.")

def train_model_on_vertex_ai(project_id, region, model_display_name, training_container_uri):
    aiplatform.init(project=project_id, location=region)
    
    job = aiplatform.CustomTrainingJob(
        display_name=model_display_name,
        script_path="train_models.py",
        container_uri=training_container_uri,
        requirements=["tensorflow==2.6.0", "numpy==1.19.5"],
    )
    
    model = job.run(
        model_display_name=model_display_name,
        replica_count=1,
        machine_type="n1-standard-4",
        accelerator_type="NVIDIA_TESLA_K80",
        accelerator_count=1,
    )
    
    print(f"Model {model.display_name} trained on Vertex AI.")
    return model

def main():
    # Upload pre-trained model to Google Cloud Storage
    bucket_name = "your-bucket-name"
    source_file_name = "models/Chess_model.h5"
    destination_blob_name = "models/Chess_model.h5"
    upload_model_to_gcs(bucket_name, source_file_name, destination_blob_name)
    
    # Train model on Vertex AI
    project_id = "your-project-id"
    region = "us-central1"
    model_display_name = "AlphaZero_Chess"
    training_container_uri = "gcr.io/cloud-aiplatform/training/tf-cpu.2-6:latest"
    trained_model = train_model_on_vertex_ai(project_id, region, model_display_name, training_container_uri)
    
    print("Cloud integration complete.")

if __name__ == "__main__":
    main()
