import googleapiclient.discovery
from google.api_core.client_options import ClientOptions
import os
import json
import base64
#import tensorflow as tf
import tensorflow.compat.v1 as tf

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./AER 1515 TPU-cefc100fc034.json"


def buildPredictJsonQuery(filePath):
    IMAGE_URI = filePath
    with tf.gfile.Open(IMAGE_URI, 'rb') as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

    image_bytes = {"b64": str(encoded_string)}
    instances = {"encoded_image": image_bytes, "key": "1"}
    with open("prediction_instances.json", "w") as f:
        f.write(json.dumps(instances))
    return instances



def predict_json(project, region, model, instances, version=None):
    """Send json data to a deployed model for prediction.

    Args:
        project (str): project where the Cloud ML Engine Model is deployed.
        region (str): regional endpoint to use; set to None for ml.googleapis.com
        model (str): model name.
        instances ([Mapping[str: Any]]): Keys should be the names of Tensors
            your deployed model expects as inputs. Values should be datatypes
            convertible to Tensors, or (potentially nested) lists of datatypes
            convertible to tensors.
        version: str, version of the model to target.
    Returns:
        Mapping[str: any]: dictionary of prediction results defined by the
            model.
    """
    # Create the ML Engine service object.
    # To authenticate set the environment variable
    # GOOGLE_APPLICATION_CREDENTIALS=<path_to_service_account_file>
    prefix = "{}-ml".format(region) if region else "ml"
    api_endpoint = "https://{}.googleapis.com".format(prefix)
    client_options = ClientOptions(api_endpoint=api_endpoint)
    service = googleapiclient.discovery.build(
        'ml', 'v1', client_options=client_options)
    name = 'projects/{}/models/{}'.format(project, model)

    if version is not None:
        name += '/versions/{}'.format(version)

    response = service.projects().predict(
        name=name,
        body={'instances': [instances]}
    ).execute()

    if 'error' in response:
        raise RuntimeError(response['error'])

    return response['predictions']


instance = buildPredictJsonQuery("test/200/001__2 Dollars_canada.jpg")
print({'instances': [instance]})
response = predict_json("aer-1515-tpu", "us-central1", "testModel", instance, "v1")
print(response)
