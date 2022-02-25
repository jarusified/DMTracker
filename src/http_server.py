import os
import warnings

from flask import Flask, json, jsonify, request
from flask_cors import CORS, cross_origin

from logger import get_logger
from interfaces import *
from generators import *

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

    def __init__(self, args):
        LOGGER.info(f"{type(self).__name__} mode enabled.")
        self.data_dir = os.path.abspath(args.args['data_dir'])
        self.experiments = os.listdir(self.data_dir)
        self.handle_routes()

    def load(self) -> None:
        
        CCT(data_dir=self.data_dir)
        H2DCudaMemcpyCommMatrixGenerator(1)

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

        @app.route("/fetchExperiments", methods=["GET"])
        @cross_origin()
        def fetch_experiments():
            return jsonify(experiments=self.experiments)

        # Example GET and POST request.
        @app.route("/fetchCCT", methods=["POST"])
        @cross_origin()
        def fetchData():
            request_context = request.json
            # base_data = readData(request_context["base"])
            # target_data = readData(request_context["target"])
            # return jsonify(base=base_data, target=target_data)
            return jsonify()
    
        