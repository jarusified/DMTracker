import argparse
from logger import get_logger

LOGGER = get_logger(__name__)


class ArgParser:
    """
    Argparser class decodes the arguments passed to
    """

    def __init__(self, args_string):

        assert isinstance(args_string, list)

        # Parse the arguments passed.
        self.parser = ArgParser._create_parser()
        self.args = vars(self.parser.parse_args())

        # Verify if only valid things are passed.
        self.mode = self._verify_parser()
        LOGGER.info(f"Mode: {self.mode}")

    def __str__(self):
        items = ("%s = %r" % (k, v) for k, v in self.__dict__.items())
        return "<%s: {%s}> \n" % (self.__class__.__name__, ", ".join(items))

    def __repr__(self):
        return self.__str__()

    # --------------------------------------------------------------------------
    # Private methods.
    @staticmethod
    def _create_parser():
        """
        Parse the input arguments.
        """
        parser = argparse.ArgumentParser(prefix_chars="--")
        parser.add_argument('--http', help="Server mode -- Spawns a http server on specified port", type=int, required=False)
        parser.add_argument('--cmd', help='input command as a string such as: -i "./app --foo 20 --bar 5"', required=False)
        parser.add_argument('--num_gpus', help='Number of gpus', type=int, default=1, required=False)
        parser.add_argument('--app_name', help='Application Name (Required)', type=str, required=True)
        parser.add_argument('--data_dir', help='Data directory path', type=str, required=False)
        return parser

    def _verify_parser(self):
        """
        Verify the input arguments.

        Raises expections if something is not provided
        Check if the config file is provided and exists!

        :pargs : argparse.Namespace
            Arguments passed by the user.

        Returns
        -------
        """

        _has_cmd = self.args["cmd"] is not None
        _has_http = self.args["http"] is not None
        _has_data_dir = self.args["data_dir"] is not None

        if not _has_data_dir:
            LOGGER.error(f'Option --data_dir not provided.')
            self.parser.print_help()
            exit(1)

        if not _has_cmd and not _has_http:
            s = "Please choose an option: --cmd or --http."
            LOGGER.error(s)
            self.parser.print_help()
            exit(1)

        if _has_http:
            mode = "HTTP"

        if _has_cmd:
            mode = "TRACE"
           
        LOGGER.info(f"{mode} triggered")
        return mode
