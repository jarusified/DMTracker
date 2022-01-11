import os
import warnings

from flask import Flask, json, jsonify, request
from flask_cors import CORS, cross_origin

from logger import get_logger

from autotm_views import readData, remapTarget
from cachedarrays_views import CachedArrayViews

# Globals
FOLDER_PATH = os.path.abspath(os.path.dirname(__file__))
STATIC_FOLDER_PATH = os.path.join(FOLDER_PATH, "app/")
LOGGER = get_logger(__name__)


# Create a Flask server.
app = Flask(__name__, static_url_path="", static_folder=STATIC_FOLDER_PATH)

# Enable CORS
cors = CORS(app, automatic_options=True)
app.config["CORS_HEADERS"] = "Content-Type"


class HTTPServer:
    """
    HTTP Server Class.
    """

    def __init__(self):

        LOGGER.info(f"{type(self).__name__} mode enabled.")
        self.data_dir = os.path.abspath("./data")
        self.handle_routes()

    def start(self, host: str, port: int) -> None:
        """
        Launch the Flask application.

        :param host: host to run API server
        :param port: port to run API server
        :return: None
        """
        LOGGER.info("Starting the API service")
        app.run(host=host, port=port, threaded=True, debug=True)

    @staticmethod
    def emit_json(endpoint: str, json_data: any) -> str:
        """
        Emit the json data to the endpoint

        :param endpoint: Endpoint to emit information to.
        :param json_data: Data to emit to the endpoint
        :return response: Response packed with data (in JSON format).
        """
        try:
            response = app.response_class(
                response=json.dumps(json_data),
                status=200,
                mimetype="application/json",
            )
            response.headers.add("Access-Control-Allow-Headers", "*")
            response.headers.add("Access-Control-Allow-Methods", "*")
            return response
        except ValueError:
            warnings.warn(f"[API: {endpoint}] emits no data.")
            return jsonify(isError=True, message="Error", statusCode=500)

    def handle_routes(self):
        @app.route("/")
        @cross_origin()
        def index():
            return app.send_static_file("index.html")

        # Example GET and POST request.
        @app.route("/fetchBase", methods=["POST"])
        @cross_origin()
        def fetchBase():
            request_context = request.json
            base_data = readData(request_context["base"])
            target_data = readData(request_context["target"])
            base_data, remap_data = remapTarget(base_data, target_data)
            return jsonify(base=base_data, target=target_data, remap=remap_data)
    
        @app.route("/fetchDatasets", methods=["POST"])
        @cross_origin()
        def fetchDatasets():
            directory = request.json["directory_name"]
            dir_files = os.listdir(os.path.join(self.data_dir, directory))
            return HTTPServer.emit_json("fetchDatasets", { "datasets": dir_files, "selectedDataset": dir_files[0] }) 