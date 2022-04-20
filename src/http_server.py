import os
import warnings

from flask import Flask, json, jsonify, request, send_from_directory
from flask_cors import CORS, cross_origin

from logger import get_logger
from interfaces import *
from generators import *

# Globals
FOLDER_PATH = os.path.abspath(os.path.dirname(__file__))
STATIC_FOLDER_PATH = os.path.join(FOLDER_PATH, "static")
LOGGER = get_logger(__name__)


# Create a Flask server.
app = Flask(__name__, static_url_path='/static')

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
        
        # Check if the directory exists. 
        HTTPServer._check_data_dir_exists(self.data_dir)

        self.experiments = os.listdir(self.data_dir)
        self.handle_routes()

    @staticmethod
    def _check_data_dir_exists(data_dir):
        """
        Internal method to check if the data_dir exists. If not, raise an
        exception and exit the program. 
        """
        _is_dir = os.path.exists(data_dir)

        if(not _is_dir):
            message = f'It looks like {data_dir} has not been created. Please run `python main.py` with --cmd option'
            LOGGER.error(message)
            exit(1) 

    
    def load(self) -> None:
        self.metrics_interface = Metrics(data_dir=self.data_dir)
        self.cct_interface = CCT(data_dir=self.data_dir)
        # self.timeline_interface = Timeline(data_dir=self.data_dir)

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

        @app.route("/fetch_experiments", methods=["GET"])
        @cross_origin()
        def fetch_experiments():
            sorted_experiments = self.metrics_interface.sort_by_runtime(self.experiments)
            return jsonify(experiments=list(sorted_experiments.keys()))

        @app.route("/fetch_cct", methods=["POST"])
        @cross_origin()
        def fetch_cct():
            request_context = request.json
            experiment = request_context["experiment"]
            nxg = self.cct_interface.get_nxg(experiment)
            return jsonify(nxg)

        @app.route("/fetch_metrics", methods=["GET"])
        @cross_origin()
        def fetch_metrics():
            metrics = self.metrics_interface.get_metrics()
            return jsonify({'metrics': metrics})

        @app.route("/fetch_kernels", methods=["POST"])
        @cross_origin()
        def fetch_kernels():
            kernels = self.metrics_interface.get_kernels()
            return jsonify({'kernels': kernels})

        @app.route("/fetch_timeline", methods=["POST"])
        @cross_origin()
        def fetch_timeline():
            request_context = request.json
            experiment = request_context["experiment"]
            timeline = self.timeline_interface.get_timeline(experiment)
            return jsonify(timeline)

        @app.route("/fetch_ensemble", methods=["POST"])
        @cross_origin()
        def fetch_ensemble():
            request_context = request.json
            metric = request_context['metric']
            if len(metric) > 0:
                data = self.metrics_interface.get_data(metric)
                return jsonify(data)

        @app.route('/static/<filename>', methods=['GET'])
        def get_json(filename):
            return send_from_directory(STATIC_FOLDER_PATH, filename)
