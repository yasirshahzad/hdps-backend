from google.cloud import storage
client = storage.Client()

bucket = client.get_bucket('images-storage-fyp-comsats')

def upload_blob(name, file):
    """Uploads a file to the bucket."""

    blob = bucket.blob(name)
    blob.upload_from_file(file)
    blob.make_public()
    
    return blob.media_link


def get_file_link(file_name): 
    blob = bucket.get_blob(file_name)
    return blob.media_link    
